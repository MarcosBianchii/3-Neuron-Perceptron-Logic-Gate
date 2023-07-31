#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

typedef struct Model {
    double u[3];
    double v[3];
    double w[3];
} Model;

double set[][3] = {
    {0,0,0},
    {0,1,1},
    {1,0,1},
    {1,1,0},
};

const unsigned int set_len = 4;

double sigmoid(double x) {
    return 1 / (1 + 1 / exp(x));
}

double sigmoid_der(double x) {
    double calc = sigmoid(x);
    return calc * (1 - calc);
}

// sigmoid(w0 sigmoid(u0 x1 + u1 x2 + u2) + w1 sigmoid(v0 x1 + v1 x2 + v2) + w2)
double pred(Model m, double x1, double x2) {
    double a = sigmoid(m.u[0] * x1 + m.u[1] * x2 + m.u[2]);
    double b = sigmoid(m.v[0] * x1 + m.v[1] * x2 + m.v[2]);
    return sigmoid(m.w[0] * a + m.w[1] * b + m.w[2]);
}

// sigmoid_der(w0 sigmoid(u0 x1 + u1 x2 + u2) + w1 sigmoid(v0 x1 + v1 x2 + v2) + w2) w0 sigmoid_der(u0 x1 + u1 x2 + u2) x1
double pred_der_u0(Model m, double x1, double x2) {
    double a = sigmoid_der(m.w[0] * sigmoid(m.u[0] * x1 + m.u[1] * x2 + m.u[2]) + m.w[1] * sigmoid(m.v[0] * x1 + m.v[1] * x2 + m.v[2]) + m.w[2]);
    return a * m.w[0] * sigmoid_der(m.u[0] * x1 + m.u[1] * x2 + m.u[2]) * x1;
}

double cost_der_u0(Model m) {
    double sum = 0;
    for (int i = 0; i < set_len; i++) {
        double x1 = set[i][0];
        double x2 = set[i][1];
        double rv = set[i][2];
        double y  = pred(m, x1, x2);
        sum += (y - rv) * pred_der_u0(m, x1, x2);
    }

    return 2 * sum / set_len;
}

// sigmoid_der(w0 sigmoid(u0 x1 + u1 x2 + u2) + w1 sigmoid(v0 x1 + v1 x2 + v2) + w2) w0 sigmoid_der(u0 x1 + u1 x2 + u2) x2
double pred_der_u1(Model m, double x1, double x2) {
    double a = sigmoid_der(m.w[0] * sigmoid(m.u[0] * x1 + m.u[1] * x2 + m.u[2]) + m.w[1] * sigmoid(m.v[0] * x1 + m.v[1] * x2 + m.v[2]) + m.w[2]);
    return a * m.w[0] * sigmoid_der(m.u[0] * x1 + m.u[1] * x2 + m.u[2]) * x2;
}

double cost_der_u1(Model m) {
    double sum = 0;
    for (int i = 0; i < set_len; i++) {
        double x1 = set[i][0];
        double x2 = set[i][1];
        double rv = set[i][2];
        double y = pred(m, x1, x2);
        sum += (y - rv) * pred_der_u1(m, x1, x2);
    }

    return 2 * sum / set_len;    
}

// sigmoid_der(w0 sigmoid(u0 x1 + u1 x2 + u2) + w1 sigmoid(v0 x1 + v1 x2 + v2) + w2) w0 sigmoid_der(u0 x1 + u1 x2 + u2)
double pred_der_u2(Model m, double x1, double x2) {
    double a = sigmoid_der(m.w[0] * sigmoid(m.u[0] * x1 + m.u[1] * x2 + m.u[2]) + m.w[1] * sigmoid(m.v[0] * x1 + m.v[1] * x2 + m.v[2]) + m.w[2]);
    return a * m.w[0] * sigmoid_der(m.u[0] * x1 + m.u[1] * x2 + m.u[2]);
}

double cost_der_u2(Model m) {
    double sum = 0;
    for (int i = 0; i < set_len; i++) {
        double x1 = set[i][0];
        double x2 = set[i][1];
        double rv = set[i][2];
        double y  = pred(m, x1, x2);
        sum += (y - rv) * pred_der_u2(m, x1, x2);
    }

    return 2 * sum / set_len;     
}

// sigmoid_der(w0 sigmoid(u0 x1 + u1 x2 + u2) + w1 sigmoid(v0 x1 + v1 x2 + v2) + w2) w1 sigmoid_der(v0 x1 + v1 x2 + v2) x1
double pred_der_v0(Model m, double x1, double x2) {
    double a = sigmoid_der(m.w[0] * sigmoid(m.u[0] * x1 + m.u[1] * x2 + m.u[2]) + m.w[1] * sigmoid(m.v[0] * x1 + m.v[1] * x2 + m.v[2]) + m.w[2]);
    return a * m.w[1] * sigmoid_der(m.v[0] * x1 + m.v[1] * x2 + m.v[2]) * x1;
}

double cost_der_v0(Model m) {
    double sum = 0;
    for (int i = 0; i < set_len; i++) {
        double x1 = set[i][0];
        double x2 = set[i][1];
        double rv = set[i][2];
        double y  = pred(m, x1, x2);
        sum += (y - rv) * pred_der_v0(m, x1, x2);
    }

    return 2 * sum / set_len;
}

// sigmoid_der(w0 sigmoid(u0 x1 + u1 x2 + u2) + w1 sigmoid(v0 x1 + v1 x2 + v2) + w2) w1 sigmoid_der(v0 x1 + v1 x2 + v2) x2
double pred_der_v1(Model m, double x1, double x2) {
    double a = sigmoid_der(m.w[0] * sigmoid(m.u[0] * x1 + m.u[1] * x2 + m.u[2]) + m.w[1] * sigmoid(m.v[0] * x1 + m.v[1] * x2 + m.v[2]) + m.w[2]);
    return a * m.w[1] * sigmoid_der(m.v[0] * x1 + m.v[1] * x2 + m.v[2]) * x2;
}

double cost_der_v1(Model m) {
    double sum = 0;
    for (int i = 0; i < set_len; i++) {
        double x1 = set[i][0];
        double x2 = set[i][1];
        double rv = set[i][2];
        double y  = pred(m, x1, x2);
        sum += (y - rv) * pred_der_v1(m, x1, x2);
    }

    return 2 * sum / set_len;      
}

// sigmoid_der(w0 sigmoid(u0 x1 + u1 x2 + u2) + w1 sigmoid(v0 x1 + v1 x2 + v2) + w2) w1 sigmoid_der(v0 x1 + v1 x2 + v2)
double pred_der_v2(Model m, double x1, double x2) {
    double a = sigmoid_der(m.w[0] * sigmoid(m.u[0] * x1 + m.u[1] * x2 + m.u[2]) + m.w[1] * sigmoid(m.v[0] * x1 + m.v[1] * x2 + m.v[2]) + m.w[2]);
    return a * m.w[1] * sigmoid_der(m.v[0] * x1 + m.v[1] * x2 + m.v[2]);
}

double cost_der_v2(Model m) {
    double sum = 0;
    for (int i = 0; i < set_len; i++) {
        double x1 = set[i][0];
        double x2 = set[i][1];
        double rv = set[i][2];
        double y  = pred(m, x1, x2);
        sum += (y - rv) * pred_der_v2(m, x1, x2);
    }

    return 2 * sum / set_len;
}

// sigmoid_der(w0 sigmoid(u0 x1 + u1 x2 + u2) + w1 sigmoid(v0 x1 + v1 x2 + v2) + w2) sigmoid(u0 x1 + u1 x2 + u2)
double pred_der_w0(Model m, double x1, double x2) {
    double a = sigmoid_der(m.w[0] * sigmoid(m.u[0] * x1 + m.u[1] * x2 + m.u[2]) + m.w[1] * sigmoid(m.v[0] * x1 + m.v[1] * x2 + m.v[2]) + m.w[2]);
    return a * sigmoid(m.u[0] * x1 + m.u[1] * x2 + m.u[2]);
}

double cost_der_w0(Model m) {
    double sum = 0;
    for (int i = 0; i < set_len; i++) {
        double x1 = set[i][0];
        double x2 = set[i][1];
        double rv = set[i][2];
        double y  = pred(m, x1, x2);
        sum += (y - rv) * pred_der_w0(m, x1, x2);
    }

    return 2 * sum / set_len;
}

// sigmoid_der(w0 sigmoid(u0 x1 + u1 x2 + u2) + w1 sigmoid(v0 x1 + v1 x2 + v2) + w2) sigmoid(v0 x1 + v1 x2 + v2)
double pred_der_w1(Model m, double x1, double x2) {
    double a = sigmoid_der(m.w[0] * sigmoid(m.u[0] * x1 + m.u[1] * x2 + m.u[2]) + m.w[1] * sigmoid(m.v[0] * x1 + m.v[1] * x2 + m.v[2]) + m.w[2]);
    return a * sigmoid(m.v[0] * x1 + m.v[1] * x2 + m.v[2]);
}

double cost_der_w1(Model m) {
    double sum = 0;
    for (int i = 0; i < set_len; i++) {
        double x1 = set[i][0];
        double x2 = set[i][1];
        double rv = set[i][2];
        double y  = pred(m, x1, x2);
        sum += (y - rv) * pred_der_w1(m, x1, x2);
    }

    return 2 * sum / set_len;
}

// sigmoid_der(w0 sigmoid(u0 x1 + u1 x2 + u2) + w1 sigmoid(v0 x1 + v1 x2 + v2) + w2)
double pred_der_w2(Model m, double x1, double x2) {
    return sigmoid_der(m.w[0] * sigmoid(m.u[0] * x1 + m.u[1] * x2 + m.u[2]) + m.w[1] * sigmoid(m.v[0] * x1 + m.v[1] * x2 + m.v[2]) + m.w[2]);
}

double cost_der_w2(Model m) {
    double sum = 0;
    for (int i = 0; i < set_len; i++) {
        double x1 = set[i][0];
        double x2 = set[i][1];
        double rv = set[i][2];
        double y  = pred(m, x1, x2);
        sum += (y - rv) * pred_der_w2(m, x1, x2);
    }

    return 2 * sum / set_len;
}

double cost(Model m) {
    double sum = 0;
    for (int i = 0; i < set_len; i++) {
        double x1 = set[i][0];
        double x2 = set[i][1];
        double rv = set[i][2];
        double y  = pred(m, x1, x2);
        double diff = y - rv;
        sum += diff * diff;
    }

    return sum / set_len;
}

void print_params_cost(Model m, double cost) {
    printf("u0 = %.2f, u1 = %.2f, u2 = %.2f, v0 = %.2f, v1 = %.2f, v2 = %.2f, w0 = %.2f, w1 = %.2f, w2 = %.2f, cost = %lf\n",
        m.u[0], m.u[1], m.u[2], m.v[0], m.v[1], m.v[2], m.w[0], m.w[1], m.w[2], cost);
}

double new_param() {
    return (double)rand() / (double)RAND_MAX;
}

int main() {
    srand(time(0));
    Model m = {
        .u = {new_param(), new_param(), new_param()},
        .v = {new_param(), new_param(), new_param()},
        .w = {new_param(), new_param(), new_param()},
    };
 
    double learning_rate = 10e-2;
    double eps = 10e-5;
    size_t max_iter = 10e+5;
    size_t iters = 0;
    double c = eps;

    do {
        m.u[0] -= cost_der_u0(m) * learning_rate;
        m.u[1] -= cost_der_u1(m) * learning_rate;
        m.u[2] -= cost_der_u2(m) * learning_rate;
        m.v[0] -= cost_der_v0(m) * learning_rate;
        m.v[1] -= cost_der_v1(m) * learning_rate;
        m.v[2] -= cost_der_v2(m) * learning_rate;
        m.w[0] -= cost_der_w0(m) * learning_rate;
        m.w[1] -= cost_der_w1(m) * learning_rate;
        m.w[2] -= cost_der_w2(m) * learning_rate;
        print_params_cost(m, c);
    } while ((c = cost(m)) > eps && ++iters != max_iter);

    printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
    for (int i = 0; i < set_len; i++)
        printf("%.0f | %.0f = %lf\n", set[i][0], set[i][1], pred(m, set[i][0], set[i][1]));
    printf("Iters: %lu\n", iters);

    return 0;
}
