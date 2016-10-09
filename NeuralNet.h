#ifndef _NEURAL_NET_H_
#define _NEURAL_NET_H_

#include "Array.h"

#include <iostream>

typedef int index;

typedef struct
{
	double value;
} Neuron;

typedef struct
{
	double weight;
	double bias;
} Sinapse;

class NeuralNet
{
public:
	NeuralNet(size_t numInputs, size_t numOutputs, size_t numPerHiddenLayer, size_t numHiddenLayers);
	virtual ~NeuralNet();

	size_t getNumNeurons(index layer);
	index getFirstNeuronIndex(index layer);
	index getLayer(index neuronIndex);
	index getPositionInLayer(index neuronIndex);

	size_t getNumInputs(index neuronIndex);
	size_t getNumOutputs(index neuronIndex);
	Sinapse* getInSinapse(index neuronIndex, index n);
	Sinapse* getOutSinapse(index neuronIndex, index n);
	Neuron* getInNeuron(index neuronIndex, index n);
	Neuron* getOutNeuron(index neuronIndex, index n);
	
	void randomizeWeights();
	void setWeights(Array<double>& weights);

	void forwardPropagate(Array<double>& inputValues, Array<double>& outputValues);

	virtual backpropagate(Array<double>& inputValues, Array<double>& outputValues);

	// TODO: make private and friendly to Tests
	index getInSinapseIndex(index neuronIndex, index n);
	index getOutSinapseIndex(index neuronIndex, index n);
	index getInNeuronIndex(index neuronIndex, index n);
	index getOutNeuronIndex(index neuronIndex, index n);

	Neuron* updateNeuronValue(index neuronIdx, size_t numInputs);
	
private:
	size_t m_numInputs;
	size_t m_numOutputs;
	size_t m_numPerHiddenLayer;
	size_t m_numHiddenLayers;
	size_t m_numNeurons;

	Array<Neuron> m_neurons;
	Array<Sinapse> m_sinapses;	
};

#endif

