#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>
#include <inttypes.h>
#include <endian.h>
#include <assert.h>
#include <math.h>
#include <random>

using namespace std;


std::random_device rd;
std::mt19937 e2(rd());
std::uniform_real_distribution<> dist(0, 1);

template <int num_bytes>
inline size_t write_be(FILE *output, void *value);
template <>
inline size_t write_be<1>(FILE *output, void *value) {
  return fwrite((char *)value, 1, 1, output);
}
template <>
inline size_t write_be<2>(FILE *output, void *value) {
  uint16_t tmp = htobe16(*(uint16_t *)value);
  return fwrite((char *)&tmp, sizeof(tmp), 1, output);
}
template <>
inline size_t write_be<4>(FILE *output, void *value) {
  uint32_t tmp = htobe32(*(uint32_t *)value);
  return fwrite((char *)&tmp, sizeof(tmp), 1, output);
}
template <>
inline size_t write_be<8>(FILE *output, void *value) {
  uint64_t tmp = htobe64(*(uint64_t *)value);
  return fwrite((char *)&tmp, sizeof(tmp), 1, output);
}

/**
 * a handy way to serialize values of certain type in big endian
 */
template <typename T>
inline size_t write_be(FILE *output, T value) {
  return write_be<sizeof(T)>(output, &value);
}

#define write_be_or_die(args...) assert(write_be(args))


struct Weight
{
    uint64_t weightId;
    uint8_t isFixed;
    double initialValue;
};

struct Variable
{
    uint64_t variableId;
    uint8_t isEvidence;
    uint64_t initialValue;
    uint16_t dataType;
    uint64_t cardinality;
};

struct VariableReference
{
    uint64_t variableId;
    uint64_t equalPredicate;
};

struct WeightReference
{
    uint64_t weightId;
    double featureValue;
};

struct Factor
{
    uint16_t factorFunction;
    uint64_t arity;
    vector<VariableReference> variableReferences;
    WeightReference weightReferences;
};

void write(vector<Variable> variable, vector<Weight> weight, vector<Factor> factor)
{
    size_t numEdges = 0;
    for (Factor f : factor) {
        numEdges += f.arity;
    }
    FILE *file = fopen("graph.meta", "w");
    fprintf(file, "%zu,%zu,%zu,%zu", weight.size(), variable.size(), factor.size(), numEdges);
    fclose(file);

    file = fopen("graph.weights", "wb");
    for (Weight w : weight) {
        write_be_or_die(file, w.weightId);
        write_be_or_die(file, w.isFixed);
        write_be_or_die(file, w.initialValue);
    }
    fclose(file);

    file = fopen("graph.variables", "wb");
    for (Variable v : variable) {
        write_be_or_die(file, v.variableId);
        write_be_or_die(file, v.isEvidence);
        write_be_or_die(file, v.initialValue);
        write_be_or_die(file, v.dataType);
        write_be_or_die(file, v.cardinality);
    }
    fclose(file);

    file = fopen("graph.factors", "wb");
    for (Factor f : factor) {
        write_be_or_die(file, f.factorFunction);
        write_be_or_die(file, f.arity);
        assert(f.arity == f.variableReferences.size());
        for (VariableReference vr : f.variableReferences) {
            write_be_or_die(file, vr.variableId);
            write_be_or_die(file, vr.equalPredicate);
        }
        write_be_or_die(file, f.weightReferences.weightId);
        write_be_or_die(file, f.weightReferences.featureValue);
    }
    fclose(file);

}

int main(int argc, char *argv[])
{
    // Ising
    /*
    const size_t N = 1000;
    const size_t M = 1000;
    const double WEIGHT = 0.1;

    vector<Variable> variable;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            Variable v;
            v.variableId = i * M + j;
            v.isEvidence = 0;
            v.initialValue = 0;
            v.dataType = 0;
            v.cardinality = 2;
            variable.push_back(v);
        }
    }

    // Option to use single weight, or one weight per factor
    vector<Weight> weight;
    Weight w;
    w.weightId = 0;
    w.isFixed = 1;
    w.initialValue = WEIGHT;
    weight.push_back(w);

    vector<Factor> factor;
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < M; j++) {
            if (i != 0) {
                Factor f;
                f.factorFunction = 3;
                f.arity = 2;
                VariableReference v1;
                VariableReference v2;
                v1.variableId = i * M + j;
                v1.equalPredicate = 0;
                v2.variableId = (i - 1) * M + j;
                v2.equalPredicate = 0;
                f.variableReferences.push_back(v1);
                f.variableReferences.push_back(v2);
                f.weightReferences.weightId = 0;
                f.weightReferences.featureValue = 1;
                factor.push_back(f);
            }
            if (j != 0) {
                Factor f;
                f.factorFunction = 3;
                f.arity = 2;
                VariableReference v1;
                VariableReference v2;
                v1.variableId = i * M + j;
                v1.equalPredicate = 0;
                v2.variableId = i * M + (j - 1);
                v2.equalPredicate = 0;
                f.variableReferences.push_back(v1);
                f.variableReferences.push_back(v2);
                f.weightReferences.weightId = 0;
                f.weightReferences.featureValue = 1;
                factor.push_back(f);
            }
        }
    }

    write(variable, weight, factor);
    */

    const size_t N = 1000;
    const double a = 1.0;
    const double b = 1.0;
    const double c = 0.5;

    vector<Weight> weight;

    Weight w;
    w.weightId = 0;
    w.isFixed = 0;
    w.initialValue = 0;
    weight.push_back(w);

    w.weightId = 1;
    w.isFixed = 0;
    w.initialValue = 0;
    weight.push_back(w);

    w.weightId = 2;
    w.isFixed = 0;
    w.initialValue = 0;
    weight.push_back(w);

    long double Z[4];
    Z[0] = exp(-a + -b +  c); // 00
    Z[1] = exp(-a +  b + -c); // 01
    Z[2] = exp( a + -b + -c); // 10
    Z[3] = exp( a +  b +  c); // 11
    cout << "Z[0]: " << Z[0] << "\n";
    cout << "Z[1]: " << Z[1] << "\n";
    cout << "Z[2]: " << Z[2] << "\n";
    cout << "Z[3]: " << Z[3] << "\n";

    Z[1] += Z[0];
    Z[2] += Z[1];
    Z[3] += Z[2];
    vector<Variable> variable;
    vector<Factor> factor;
    int count[4];
    count[0] = 0;
    count[1] = 0;
    count[2] = 0;
    count[3] = 0;
    for (size_t i = 0; i < N; i++) {
        long double r = dist(e2) * Z[3];

        uint32_t index;
        if (r < Z[0]) {
            index = 0;
        }
        else if (r < Z[1]) {
            index = 1;
        }
        else if (r < Z[2]) {
            index = 2;
        }
        else {
            index = 3;
        }
        count[index]++;
        cout << index << "\n";

        Variable v;

        v.variableId = 2 * i;
        v.isEvidence = 1;
        v.initialValue = (index == 2 || index == 3);
        v.dataType = 0;
        v.cardinality = 2;
        variable.push_back(v);

        v.variableId = 2 * i + 1;
        v.isEvidence = 1;
        v.initialValue = (index == 1 || index == 3);
        v.dataType = 0;
        v.cardinality = 2;
        variable.push_back(v);


        VariableReference v1;
        v1.variableId = 2 * i;
        v1.equalPredicate = 0;
        VariableReference v2;
        v2.variableId = 2 * i + 1;
        v2.equalPredicate = 0;

        Factor f1;
        f1.factorFunction = 4;
        f1.arity = 1;
        f1.variableReferences.push_back(v1);
        f1.weightReferences.weightId = 0;
        f1.weightReferences.featureValue = 1;
        factor.push_back(f1);

        Factor f2;
        f2.factorFunction = 4;
        f2.arity = 1;
        f2.variableReferences.push_back(v2);
        f2.weightReferences.weightId = 1;
        f2.weightReferences.featureValue = 1;
        factor.push_back(f2);

        Factor f3;
        f3.factorFunction = 3;
        f3.arity = 2;
        f3.variableReferences.push_back(v1);
        f3.variableReferences.push_back(v2);
        f3.weightReferences.weightId = 2;
        f3.weightReferences.featureValue = 1;
        factor.push_back(f3);
    }
    cout << count[0] << "\n";;
    cout << count[1] << "\n";;
    cout << count[2] << "\n";;
    cout << count[3] << "\n";;

    write(variable, weight, factor);

    return 0;

}

