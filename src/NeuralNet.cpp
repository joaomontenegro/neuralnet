
#include "NeuralNet.h"

#include <stdlib.h>
#include <time.h>

NeuralNet::NeuralNet(Array<size_t>& neuronsPerLayer)
{


	size_t numLayers = neuronsPerLayer.size();

	m_neurons.allocate(numLayers);
	m_sinapses.allocate(numLayers - 1);

	for (size_t l = 0; l < numLayers; ++l)
	{
		size_t numNeurons = neuronsPerLayer[l];
		m_neurons[l].allocate(numNeurons);

		if (l < numLayers - 1)
		{
			m_sinapses[l].allocate(numNeurons * neuronsPerLayer[l + 1]);
		}
	}
}

NeuralNet::~NeuralNet() { }

size_t NeuralNet::getNumLayers()
{
	return m_neurons.size();
}

size_t NeuralNet::getNumNeurons(size_t layer)
{
	return m_neurons[layer].size(); // TODO check num layers
}

void NeuralNet::getOutputs(Array<double>& values)
{
	size_t outputLayer = m_neurons.size() - 1;
	size_t numOutputs = m_neurons[outputLayer].size();

	values.allocate(numOutputs);

	for (size_t n = 0; n < numOutputs; ++n)
	{
		values[n] = m_neurons[outputLayer][n].value;
	}

	return;
}

NeuralNet::Sinapse* NeuralNet::getSinapse(size_t layer, size_t neuronIndexFrom, size_t neuronIndexTo)
{
	// TODO test sizes
	size_t index = m_neurons[layer + 1].size() * neuronIndexFrom + neuronIndexTo;
	return &m_sinapses[layer][index];
}

void NeuralNet::setSinapses(Array<Sinapse>& sinapses)
{
	//TODO test sizes

	size_t i = 0;

	for (size_t l = 0; l < m_sinapses.size(); ++l)
	{
		for (size_t s = 0; s < m_sinapses[l].size(); ++s)
		{
			m_sinapses[l][s] = sinapses[i++];
		}
	}
}

void NeuralNet::randomize()
{
	srand((unsigned int)time(NULL));

	for (size_t l = 0; l < m_sinapses.size(); ++l)
	{
		//TODO store m_sinapses[l] in var

		for (size_t s = 0; s < m_sinapses[l].size(); ++s)
		{
			//TODO store m_sinapses[l][s] in var
			m_sinapses[l][s].weight = (double)rand() / (double)(RAND_MAX * 2.0 - 1.0);
			m_sinapses[l][s].bias = (double)rand() / (double)(RAND_MAX * 2.0 - 1.0);
		}
	}
}

void NeuralNet::forwardPropagate(Array<double>& inputValues)
{
	// Set inputs
	for (size_t i = 0; i < inputValues.size(); ++i)
	{
		//TODO: check input layer size
		m_neurons[0][i].value = inputValues[i];
	}

	size_t numLayers = getNumLayers();
	size_t numPreviousNeurons = m_neurons[0].size();
	size_t numNeurons;

	// Layers after input
	for (size_t l = 1; l < numLayers; ++l)
	{
		numNeurons = m_neurons[l].size();

		// Current layer's neurons
		for (size_t n = 0; n < numNeurons; ++n) 
		{
			double value = 0;

			// Previous layer's neurons
			for (size_t p = 0; p < numPreviousNeurons; ++p)
			{
				Sinapse* sinapse = getSinapse(l - 1, p, n);
				double previousValue = m_neurons[l - 1][p].value;
				value += previousValue * sinapse->weight + sinapse->bias;
			}

			// Update neuron value
			m_neurons[l][n].value = value;
		}

		numPreviousNeurons = numNeurons;
	}
}

void NeuralNet::print()
{
	size_t numLayers = getNumLayers();

	for (size_t l = 0; l < numLayers; ++l)
	{
		std::cout << " N" << l << ": ";
		for (int n = 0 ; n < m_neurons[l].size(); ++n)
		{
			std::cout << m_neurons[l][n].value << " ";
		}

		std::cout << std::endl;

		if ( l < numLayers - 1)
		{
			std::cout << " S" << l << ": ";
			for (int s = 0 ; s < m_sinapses[l].size(); ++s)
			{
				std::cout << "(" << m_sinapses[l][s].weight << ", " << m_sinapses[l][s].bias << ") ";
			}
		}

		std::cout << std::endl;
	}
}

