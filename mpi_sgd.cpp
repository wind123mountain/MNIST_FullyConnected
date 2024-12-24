#include <Eigen/Dense>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <random>
#include <sstream>
#include <stdlib.h>
#include <string>

using namespace Eigen;

typedef Matrix<float, -1, -1, Eigen::RowMajor> MatrixXfRow;

struct dataset {
  int n;
  float *X;
  int *y;
};

class Act_Function {
public:
#define sigmoid 1

  virtual ~Act_Function() {}

  virtual MatrixXf call(MatrixXf &m) { return m; }
  virtual MatrixXf derivative(MatrixXf &m) {
    return MatrixXf::Ones(m.rows(), m.cols());
  }

  static Act_Function *get_act_funct(int funct);
};

class Sigmoid : public Act_Function {
public:
  MatrixXf call(MatrixXf &m) override {
    return (1.0f / (1.0f + (-m.array()).exp()));
  }
  MatrixXf derivative(MatrixXf &m) override {
    return m.array() * (1.0f - m.array());
  }
};

Act_Function *Act_Function::get_act_funct(int funct) {
  switch (funct) {
  case sigmoid:
    return new Sigmoid();

  default:
    return new Act_Function();
  }
}

class SparseCategoricalCrossentropy {
public:
  float call(MatrixXf &X, VectorXi &y) {
    VectorXf loss(y.size());
    VectorXf softmax;
    for (int i = 0; i < X.rows(); i++) {
      softmax = X.row(i).array().exp();
      softmax = softmax / softmax.sum();
      loss(i) = -std::log(softmax(y(i)));
    }

    return loss.sum();
  }

  MatrixXf derivative(MatrixXf &X, VectorXi &y) {
    MatrixXf d = X.array().exp();
    float sum_row;
    for (int i = 0; i < X.rows(); i++) {
      sum_row = d.row(i).sum();
      d.row(i) /= sum_row;
      d(i, y(i)) -= 1;
    }

    return d;
  }
};

class Densen_Layer {
public:
  int input_size;
  int output_size;
  MatrixXf weight;
  RowVectorXf bias;
  MatrixXf output;
  Act_Function *act;

  float *w_gra_arr;
  Map<MatrixXfRow> *w_gradient;
  float *bias_gra_arr;
  Map<RowVectorXf> *bias_gradient;

  Densen_Layer(int in_size, int out_size, int act_funct = 0) {
    input_size = in_size;
    output_size = out_size;
    weight = MatrixXf::Random(in_size, out_size);
    bias = RowVectorXf::Random(out_size);
    act = Act_Function::get_act_funct(act_funct);
    w_gra_arr = (float *)malloc(in_size * out_size * sizeof(float));
    w_gradient = new Map<MatrixXfRow>(w_gra_arr, in_size, out_size);
    bias_gra_arr = (float *)malloc(out_size * sizeof(float));
    bias_gradient = new Map<RowVectorXf>(bias_gra_arr, out_size);
  }

  ~Densen_Layer() {
    delete act;
    free(w_gra_arr);
    delete w_gradient;
    free(bias_gra_arr);
    delete bias_gradient;
  }

  MatrixXf feedforward(MatrixXf &X) {
    output = X * weight;
    output = output.rowwise() + bias;
    output = act->call(output);
    return output;
  }
};

class Network_Model {
public:
  int input_size;
  int n_label;
  int n_densen = 5;
  Densen_Layer layers[5];
  SparseCategoricalCrossentropy loss;

  Network_Model(int input_size, int n_label)
      : layers{Densen_Layer(input_size, 32, sigmoid),
               Densen_Layer(32, 64, sigmoid), Densen_Layer(64, 64, sigmoid),
               Densen_Layer(64, 32, sigmoid), Densen_Layer(32, n_label)} {
    this->n_label = n_label;
    this->input_size = input_size;
  }

  MatrixXf feedforward(MatrixXf &X) {
    MatrixXf output = X;
    for (int i = 0; i < n_densen; i++) {
      output = layers[i].feedforward(output);
    }
    return output;
  }

  void backward(MatrixXf &X, VectorXi &y) {
    MatrixXf output_delta = loss.derivative(layers[n_densen - 1].output, y);
    output_delta =
        output_delta.array() * layers[n_densen - 1]
                                   .act->derivative(layers[n_densen - 1].output)
                                   .array();

    *layers[n_densen - 1].w_gradient =
        layers[n_densen - 2].output.transpose() * output_delta;
    *layers[n_densen - 1].bias_gradient = output_delta.colwise().sum();

    for (int i = n_densen - 2; i > -1; i--) {
      output_delta = output_delta * layers[i + 1].weight.transpose();
      output_delta = output_delta.array() *
                     layers[i].act->derivative(layers[i].output).array();

      *layers[i].w_gradient =
          (i > 0 ? layers[i - 1].output.transpose() : X.transpose()) *
          output_delta;

      *layers[i].bias_gradient = output_delta.colwise().sum();
    }
  }

  VectorXi to_label(MatrixXf &predict) {
    VectorXi labels(predict.rows());
    for (int i = 0; i < predict.rows(); i++) {
      float max = predict(i, 0);
      labels(i) = 0;
      for (int j = 1; j < predict.cols(); j++) {
        if (predict(i, j) > max) {
          max = predict(i, j);
          labels(i) = j;
        }
      }
    }
    return labels;
  }

  int get_num_predict_true(VectorXi &predict_labels, VectorXi &label) {
    return (predict_labels.array() - label.array() == 0).cast<int>().sum();
  }

  void save_params(std::string path) {
    std::ofstream file(path);

    // Check if the file is open
    if (!file) {
      std::cerr << "Error opening file!" << std::endl;
      return;
    }

    for (int i = 0; i < n_densen; i++) {
      file << layers[i].weight << "\n" << layers[i].bias << std::endl;
    }

    file.close();
  }

  void load_params(std::string path) {
    std::ifstream file(path);

    // Check if the file is open
    if (!file) {
      std::cerr << "Error opening file!" << std::endl;
      return;
    }

    for (int i = 0; i < n_densen; i++) {
      for (int r = 0; r < layers[i].weight.rows(); r++) {
        for (int c = 0; c < layers[i].weight.cols(); c++) {
          file >> layers[i].weight(r, c);
        }
      }

      for (int j = 0; j < layers[i].bias.size(); j++) {
        file >> layers[i].bias(j);
      }
    }

    file.close();
  }
};

void SGD(Network_Model &model, int n_sample, float lr) {
  for (int i = 0; i < model.n_densen; i++) {
    model.layers[i].weight -= *model.layers[i].w_gradient * lr / n_sample;
    model.layers[i].bias -= *model.layers[i].bias_gradient * lr / n_sample;
  }
}

void load_data(std::string path, float *X, int *y, int size) {
  std::ifstream file(path);

  if (!file.is_open()) {
    std::cerr << "Could not open the file!" << std::endl;
    return;
  }

  std::string line;

  int i;
  for (i = 0; std::getline(file, line); i++) {
    std::stringstream ss(line);
    std::string cell;
    std::vector<std::string> row;

    std::getline(ss, cell, ',');
    y[i] = std::atoi(cell.c_str());

    // Split the line into cells based on the comma delimiter
    int j;
    for (j = 0; std::getline(ss, cell, ','); j++) {
      X[i * size + j] = (std::atoi(cell.c_str()) - 127.5) / 127.5;
    }
  }

  // std::cout << path << " " << i << " samples" << std::endl;

  file.close();
}

void train_step(Network_Model &model, int epochs, int batch_size, float lr,
                dataset train, int ncpu, int rank) {
  std::time_t start = std::time(nullptr);
  int n_batch = train.n / batch_size;
  n_batch += train.n % batch_size > 0 ? 1 : 0;
  int n_sample_last_batch = train.n % batch_size;
  n_sample_last_batch += train.n % batch_size > 0 ? 0 : batch_size;

  for (int epoch = 0; epoch < epochs; epoch++) {
    float total_loss_ms = 0;
    float acc_ms = 0;
    for (int step = 0; step < n_batch; step++) {
      int n_sample = step < n_batch - 1 ? batch_size : n_sample_last_batch;
      int n_ms = n_sample / ncpu;
      int n_mod = n_sample % ncpu;

      float *start_X = train.X + (step * batch_size + n_ms * rank) * model.input_size;
      int *start_y = train.y + step * batch_size + n_ms * rank;

      n_ms += (rank == ncpu - 1 && n_mod > 0) ? n_mod : 0;

      MatrixXf X = Map<MatrixXfRow>(start_X, n_ms, model.input_size);
      VectorXi y = Map<VectorXi>(start_y, n_ms);

      model.feedforward(X);
      model.backward(X, y);

      for (int i = 0; i < model.n_densen; i++) {
        float w_gra[model.layers[i].weight.size()];
        std::memcpy(w_gra, model.layers[i].w_gra_arr,
                    model.layers[i].weight.size() * sizeof(float));

        float b_gra[model.layers[i].bias.size()];
        std::memcpy(b_gra, model.layers[i].bias_gra_arr,
                    model.layers[i].bias.size() * sizeof(float));

        MPI_Allreduce(w_gra, model.layers[i].w_gra_arr,
                      model.layers[i].weight.size(), MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(b_gra, model.layers[i].bias_gra_arr,
                      model.layers[i].bias.size(), MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
      }

      SGD(model, n_sample, lr);

      int out_layer_id = model.n_densen - 1;
      total_loss_ms += model.loss.call(model.layers[out_layer_id].output, y);
      VectorXi predict = model.to_label(model.layers[out_layer_id].output);
      acc_ms += model.get_num_predict_true(predict, y);
    }

    float total_loss, acc;
    MPI_Reduce(&total_loss_ms, &total_loss, 1, MPI_FLOAT, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&acc_ms, &acc, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (epoch % 20 == 0 && rank == 0) {
      std::cout << "Epoch " << epoch << ": loss = " << total_loss / train.n
                << " - acc = " << acc / train.n << std::endl;
    }
  }

  if (rank == 0) {
    std::time_t end = std::time(nullptr);
    std::cout << "Train time: " << end - start << " secconds" << std::endl;
  }
}

int main(int argc, char *argv[]) {
  int ncpu, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &ncpu);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  srand(41);

  std::time_t start = std::time(nullptr);

  int input_size = 28 * 28;
  int n_label = 10;
  int n_train = 60000;
  int n_test = 10000;

  dataset train = {n_train,
                   (float *)malloc(input_size * n_train * sizeof(float)),
                   (int *)malloc(n_train * sizeof(int))};
  load_data("MNIST_CSV/mnist_train.csv", train.X, train.y, input_size);

  dataset test = {n_test, (float *)malloc(input_size * n_test * sizeof(float)),
                  (int *)malloc(n_test * sizeof(int))};
  load_data("MNIST_CSV/mnist_test.csv", test.X, test.y, input_size);

  int epochs = 100;
  int batch_size = std::atoi(argv[argc - 1]);
  float lr = 0.1;

  Network_Model model(input_size, n_label);
  // model.save_params("params");
  model.load_params("params");

  train_step(model, epochs, batch_size, lr, train, ncpu, rank);

  if (rank == 0) {
    MatrixXf X = Map<MatrixXfRow>(test.X, test.n, input_size);
    VectorXi y = Map<VectorXi>(test.y, test.n);

    model.feedforward(X);

    int out_layer_id = model.n_densen - 1;
    float loss = model.loss.call(model.layers[out_layer_id].output, y) / test.n;
    VectorXi predict = model.to_label(model.layers[out_layer_id].output);
    float acc = model.get_num_predict_true(predict, y) * 100.0 / test.n;

    std::cout << "Test: loss = " << loss << " - acc = " << acc << std::endl;

    std::time_t end = std::time(nullptr);
    std::cout << "Runtime: " << end - start << " secconds" << std::endl;
  }

  MPI_Finalize();

  return 0;
}
