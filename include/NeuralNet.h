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
	} Neuron;

	typedef struct
	{
		double weight;
		double bias;
	} Sinapse;

public:
	NeuralNet(Array<size_t>& neuronsPerLayer);
	virtual ~NeuralNet();

	size_t getNumLayers();
	size_t getNumNeurons(size_t layer);
	void getOutputs(Array<double>& values);

	void setSinapses(Array<Sinapse>& sinapses);
	void randomize();

	void forwardPropagate(Array<double>& inputValues);
	

	void print();
private:
	typedef Array<Neuron> NeuronArray; 
	typedef Array<Sinapse> SinapseArray;

	Array<NeuronArray> m_neurons;
	Array<SinapseArray> m_sinapses;

	Sinapse* getSinapse(size_t layer, size_t neuronIndexFrom, size_t neuronIndexTo);
};

#endif

