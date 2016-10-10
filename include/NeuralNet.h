#ifndef _NEURAL_NET_H_
#define _NEURAL_NET_H_

#include "Array.h"

#include <iostream>
#include <stdlib.h>

class NeuralNet
{
public: //typedefs
	typedef struct
	{
		double value;
		double bias;
	} Neuron;

	typedef struct
	{
		double weight;
	} Sinapse;

public:
	NeuralNet(Array<size_t>& neuronsPerLayer);
	virtual ~NeuralNet();

	size_t getNumLayers();
	size_t getLayerSize(size_t layer);
	void getOutputs(Array<double>& values);

	void set(Array<Neuron>& neurons, Array<Sinapse>& sinapses);
	void randomize();

	static double activationFunction(double value);
	static double activationFunctionDerivative(double value);

	double error(Array<double>& inputValues, Array<double>& outputValues);

	void forwardPropagate(Array<double>& inputValues);
	void backPropagate(Array<double>& inputValues, Array<double>& outputValues, double rate, double biasRate);
	

	void print();

private:
	typedef Array<Neuron> NeuronArray; 
	typedef Array<Sinapse> SinapseArray;

	Array<NeuronArray> m_neurons;
	Array<SinapseArray> m_sinapses;
	Array<double> m_biases; //TODO

	size_t m_maxLayerSize;
	size_t m_maxSinapsesInLayer;

	Sinapse* getSinapse(size_t layer, size_t neuronIndexFrom, size_t neuronIndexTo);
};

#endif

