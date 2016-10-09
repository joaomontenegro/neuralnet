
#include "NeuralNet.h"

#include <stdlib.h>
#include <time.h>

NeuralNet::NeuralNet(size_t numInputs, size_t numOutputs, size_t numPerHiddenLayer, size_t numHiddenLayers)
	: m_numInputs(numInputs),
	  m_numOutputs(numOutputs),
	  m_numPerHiddenLayer(numPerHiddenLayer),
	  m_numHiddenLayers(numHiddenLayers)
{
	m_numNeurons = m_numInputs + m_numOutputs + m_numHiddenLayers * m_numPerHiddenLayer;
	size_t numSinapses = numPerHiddenLayer * (numPerHiddenLayer * (numHiddenLayers - 1) + numInputs + numOutputs);
	m_neurons.allocate(m_numNeurons);
	m_sinapses.allocate(numSinapses);	
}

NeuralNet::~NeuralNet() {}

size_t NeuralNet::getNumNeurons(index layer)
{
	if (layer < 0)
	{
		return 0;
	}

	if (layer == 0)
	{
		// Input layer
		return m_numInputs;
	}
	else if (layer == m_numHiddenLayers + 1)
	{
		// Output layer
		return m_numOutputs;
	}
	else if (layer > (int)m_numHiddenLayers + 1)
	{
		// Invalid layer
		return 0;
	}
	
	// Hidden layer
	return m_numPerHiddenLayer;
}

index NeuralNet::getFirstNeuronIndex(index layer)
{
	if (layer < 0)
	{
		return -1;
	}

	if (layer == 0)
	{
		// Input layer
		return 0;
	}
	else if (layer == m_numHiddenLayers + 1)
	{
		// Output layer
		return m_numInputs + m_numHiddenLayers * m_numPerHiddenLayer;
	}
	else if (layer > (int)m_numHiddenLayers + 1)
	{
		// Invalid layer
		return -1;
	}
	
	// Hidden layer
	return m_numInputs + (layer - 1) * m_numPerHiddenLayer;
}

index NeuralNet::getLayer(index neuronIndex)
{
	if (neuronIndex < 0 || neuronIndex >= (int)m_numNeurons)
	{
		return -1;
	}

	if (neuronIndex < (int)m_numInputs)
	{
		// Input layer
		return 0;
	}
	
	index firstOutput = getFirstNeuronIndex(m_numHiddenLayers + 1);
	index firstInvalid = firstOutput + m_numOutputs;
	if (neuronIndex > firstOutput && neuronIndex < firstInvalid)
	{
		// Output layer
		return m_numHiddenLayers + 1;
	}

	if (neuronIndex >= firstInvalid)
	{
		// Invalid
		return -1;
	}

	// Hidden layer
	return 1 + (neuronIndex - m_numInputs) / m_numPerHiddenLayer;
}

index NeuralNet::getPositionInLayer(index neuronIndex)
{
	index layer = getLayer(neuronIndex);
	if (layer < 0)
	{
		return -1;
	}

	return neuronIndex - getFirstNeuronIndex(layer);
}

size_t NeuralNet::getNumInputs(index neuronIndex)
{
	index layer = getLayer(neuronIndex);

	if (layer <= 0 || layer > (int)m_numHiddenLayers + 1)
	{
		return 0;
	}

	return getNumNeurons(layer - 1);
}

size_t NeuralNet::getNumOutputs(index neuronIndex)
{
	index layer = getLayer(neuronIndex);

	if (layer < 0 || layer >= (int)m_numHiddenLayers + 1)
	{
		return 0;
	}

	return getNumNeurons(layer + 1);
}


index NeuralNet::getInSinapseIndex(index neuronIndex, index n)
{
	if (neuronIndex < (int)m_numInputs || neuronIndex >= (int)m_numNeurons)
	{
		// Input layer or invalid
		return -1;
	}

	index layer = getLayer(neuronIndex);
	index position = neuronIndex - getFirstNeuronIndex(layer);

	if (layer == 1)
	{
		// First hidden layer (inputs connected to input layer)
		if (n >= (int)m_numInputs) return -1;
		return m_numPerHiddenLayer * n + position;
	}

	// All the remaining ones should be connected to an hidden layer
	if (n >= (int)m_numPerHiddenLayer) return -1;

	if (layer <= (int)m_numHiddenLayers)
	{
		// Other hidden layers
		return m_numPerHiddenLayer * (m_numInputs + m_numPerHiddenLayer * (layer - 2) + n) + position;
	}

	// Output layer
	return m_numPerHiddenLayer * (m_numInputs + m_numPerHiddenLayer * (layer - 2)) + n * m_numOutputs + position;
}

Sinapse* NeuralNet::getInSinapse(index neuronIndex, index n)
{
	index idx = getInSinapseIndex(neuronIndex, n);
	if (idx < 0)
	{
		return 0x0;
	}

	return &m_sinapses[idx];
}

index NeuralNet::getOutSinapseIndex(index neuronIndex, index n)
{
	if (neuronIndex < 0 || neuronIndex >= (int)(m_numNeurons - m_numOutputs))
	{
		// Output Layer or invalid
		return -1;
	}

	if (neuronIndex < (int)m_numInputs)
	{
		// Input Layer
		if (n >= (int)m_numPerHiddenLayer)
		{
			return -1;
		}

		return m_numPerHiddenLayer * neuronIndex + n;
	}

	index layer = getLayer(neuronIndex);
	index position = neuronIndex - getFirstNeuronIndex(layer);

	if (layer < (int)m_numHiddenLayers)
	{
		// All hidden layers except the last one (outputs connected to hidden layers)
		if (n >= (int)m_numPerHiddenLayer) return -1;
		return (m_numInputs + (layer - 1) * m_numPerHiddenLayer + position) * m_numPerHiddenLayer + n;
	}

	// last hidden layer (connected to the output)
	if (n >= (int)m_numOutputs) return -1;
	return (m_numInputs + (layer - 1) * m_numPerHiddenLayer) * m_numPerHiddenLayer + position * m_numOutputs + n;
}

Sinapse* NeuralNet::getOutSinapse(index neuronIndex, index n)
{
	index idx = getOutSinapseIndex(neuronIndex, n);
	if (idx < 0)
	{
		return 0x0;
	}

	return &m_sinapses[idx];
}

index NeuralNet::getInNeuronIndex(index neuronIndex, index n)
{
	if (neuronIndex < (int)m_numInputs || neuronIndex >= (int)m_numNeurons)
	{
		// Input layer or invalid
		return -1;
	}

	index layer = getLayer(neuronIndex) - 1;
	
	if (n < 0 || n >= (int)getNumNeurons(layer))
	{
		// Invalid n
		return -1;
	}

	return getFirstNeuronIndex(layer) + n;
}

Neuron* NeuralNet::getInNeuron(index neuronIndex, index n)
{
	index idx = getInNeuronIndex(neuronIndex, n);
	if (idx < 0)
	{
		return 0x0;
	}

	return &m_neurons[idx];
}

index NeuralNet::getOutNeuronIndex(index neuronIndex, index n)
{
	if (neuronIndex < 0 || neuronIndex >= (int)(m_numNeurons - m_numOutputs))
	{
		// Invalid or Output layer
		return -1;
	}

	index layer = getLayer(neuronIndex) + 1;

	if (n < 0 || n >= (int)getNumNeurons(layer))
	{
		// Invalid n
		return -1;
	}

	return getFirstNeuronIndex(layer) + n;
}

Neuron* NeuralNet::getOutNeuron(index neuronIndex, index n)
{
	index idx = getOutNeuronIndex(neuronIndex, n);
	if (idx < 0)
	{
		return 0x0;
	}

	return &m_neurons[idx];
}

Neuron* NeuralNet::updateNeuronValue(index neuronIdx, size_t numInputs)
{
	double value = 0;
	for (size_t input = 0; input < numInputs; ++input)
	{
		Sinapse* sinapse = getInSinapse(neuronIdx, input);
		Neuron* inNeuron = getInNeuron(neuronIdx, input);
		value += inNeuron->value * sinapse->weight;
	}

	m_neurons[neuronIdx].value = value;

	return &m_neurons[neuronIdx];
}

void NeuralNet::setWeights(Array<double>& weights)
{
	for (size_t s = 0; s < m_sinapses.size(); ++s)
	{
		m_sinapses[s].weight = weights[s];
	}
}

void NeuralNet::randomizeWeights()
{
	srand((unsigned int)time(NULL));

	for (size_t s = 0; s < m_sinapses.size(); ++s)
	{
		double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX) * 2.0 - 1.0;
		m_sinapses[s].weight = r;
	}
}

void NeuralNet::forwardPropagate(Array<double>& inputValues, Array<double>& outputValues)
{
	// Populate input values
	for (size_t i = 0; i < m_numInputs; ++i)
	{
		m_neurons[i].value = inputValues[i];
	}

	// Propagate into hidden layers
	int numInlayer = m_numInputs;
	index neuronIdx = m_numInputs;
	for (size_t layer = 0; layer < m_numHiddenLayers; ++layer)
	{
		for (size_t n = 0; n < m_numPerHiddenLayer; ++n, ++neuronIdx)
		{
			updateNeuronValue(neuronIdx, numInlayer);
		}

		numInlayer = m_numPerHiddenLayer;
	}
	
	// Propagate into output layer
	for (size_t n = 0; n < m_numOutputs; ++n, ++neuronIdx)
	{
		Neuron* neuron = updateNeuronValue(neuronIdx, m_numPerHiddenLayer);
		outputValues[n] = neuron->value;
	}
}

