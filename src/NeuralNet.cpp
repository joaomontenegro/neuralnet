
#include "NeuralNet.h"

#include <math.h>
#include <stdlib.h>
#include <time.h>

NeuralNet::NeuralNet(Array<size_t>& neuronsPerLayer) : m_maxLayerSize(0), m_maxSinapsesInLayer(0)
{
	size_t numLayers = neuronsPerLayer.size();

	m_neurons.allocate(numLayers);
	m_sinapses.allocate(numLayers - 1);

	for (size_t l = 0; l < numLayers; ++l)
	{
		size_t numNeurons = neuronsPerLayer[l];
		m_neurons[l].allocate(numNeurons);

		// Find the largest layer
		if (numNeurons > m_maxLayerSize)
		{
			m_maxLayerSize = numNeurons;
		}

		// Allocate sinapses
		if (l < numLayers - 1)
		{
			size_t numSinapses = numNeurons * neuronsPerLayer[l + 1];
			m_sinapses[l].allocate(numSinapses);

			// Find the largest number of sinapses per layer
			if (numSinapses > m_maxSinapsesInLayer)
			{
				m_maxSinapsesInLayer = numSinapses;
			}
		}
	}
}

NeuralNet::~NeuralNet() { }

size_t NeuralNet::getNumLayers()
{
	return m_neurons.size();
}

size_t NeuralNet::getLayerSize(size_t layer)
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

void NeuralNet::set(Array<Neuron>& neurons, Array<Sinapse>& sinapses)
{
	//TODO test sizes

	size_t i = 0;

	for (size_t l = 0; l < m_neurons.size(); ++l)
	{
		for (size_t n = 0; n < m_neurons[l].size(); ++n)
		{
			m_neurons[l][n] = neurons[i++];
		}
	}

	i = 0;
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
		for (size_t s = 0; s < m_sinapses[l].size(); ++s)
		{
			m_sinapses[l][s].weight = (double)rand() / (double)(RAND_MAX * 2.0 - 1.0);
		}

		for (size_t n = 0; n < m_neurons[l].size(); ++n)
		{
			m_neurons[l][n].bias = (double)rand() / (double)(RAND_MAX * 2.0 - 1.0);
		}
	}
}

double NeuralNet::activationFunction(double value)
{
	// Sigmoid
	return 1.0 / (1.0 + powl(M_E, -value));
}

double NeuralNet::activationFunctionDerivative(double value)
{
	// Sigmoid derivative
	return value * (1.0 - value);
}

double NeuralNet::error(Array<double>& inputValues, Array<double>& outputValues)
{
	forwardPropagate(inputValues);

	// Squared error function 0.5 * sum((target - value)^2)
	double error = 0;
	size_t outputLayer = m_neurons.size() - 1;
	for (size_t i = 0; i < outputValues.size(); ++i)
	{
		error += powl(outputValues[i] - m_neurons[outputLayer][i].value, 2);
	}

	return error / 2;
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
			double bias = m_neurons[l][n].bias;

			// Previous layer's neurons
			for (size_t p = 0; p < numPreviousNeurons; ++p)
			{
				Sinapse* sinapse = getSinapse(l - 1, p, n);
				double previousValue = m_neurons[l - 1][p].value;
				value += previousValue * sinapse->weight + bias;
			}

			// Update neuron value
			m_neurons[l][n].value = activationFunction(value);
		}

		numPreviousNeurons = numNeurons;
	}
}

void NeuralNet::backPropagate(Array<double>& inputValues, Array<double>& outputValues, double rate)
{
	//tODO confirm output and input sizes
	forwardPropagate(inputValues);

	// A neuron: previousValue -> weight -> netValue - > value
	// netValue is before activation func, value is after activation func
	// We want to find the partial derivative dError/dWeight
	// dError/dWeight = dError/dValue * dValue/dNetValue * dNetValue/dWeight
	// Then weight -= rate * dError/dWeight (gradient descent)

	// When walking back the layers the dError/dNetValues will be needed several
	// times, so we cache them using an array that can fit the largest layer.
	// Because we need the current values  while we building the array for the
	// next one we will create two arrays and use two pointers that swap between
	// them on each layer iteration
	//TODO: IMPROVE ARRAY TO ALLOW FAST SWAP
	Array<double> dError_dNetValues0(m_maxLayerSize);
	Array<double> dError_dNetValues1(m_maxLayerSize);
	Array<double>* dError_dNetValues = &dError_dNetValues0;
	Array<double>* next_dError_dNetValues = &dError_dNetValues1;
	
	// Stores the weights on each iteration before the update
	Array<double> oldWeights(m_maxSinapsesInLayer);

	// We start with the output layer
	size_t outputLayer = m_neurons.size() - 1;
	size_t layerSize = getLayerSize(outputLayer);
	size_t previousLayer = outputLayer - 1;
	size_t previousLayerSize = m_neurons[outputLayer - 1].size();
	size_t sinapseIdx = 0;

	for (size_t n = 0; n < layerSize; ++n)
	{
		double value = m_neurons[outputLayer][n].value;
		double dError_dValue = value - outputValues[n];
		double dValue_dNetValue = activationFunctionDerivative(value);
		double dError_dNetValue = dError_dValue * dValue_dNetValue;

		// Cache dError/dNetValue
		(*next_dError_dNetValues)[n] = dError_dNetValue;

		for (size_t p = 0; p < previousLayerSize; ++p)
		{
			Sinapse* sinapse = getSinapse(previousLayer, p, n);
			double dError_dWeight = m_neurons[previousLayer][p].value;

			// Update Output weights
			oldWeights[sinapseIdx++] = sinapse->weight;
			sinapse->weight -= rate * dError_dNetValue * dError_dWeight;
		}
	}

	// Then the hidden layers
	for (size_t l = outputLayer - 1; l > 0; --l)
	{
		layerSize = getLayerSize(l);
		previousLayer = l - 1;
		previousLayerSize = getLayerSize(previousLayer);
		double nextLayerSize = getLayerSize(l + 1);

		sinapseIdx = 0;

		for (size_t n = 0; n < layerSize; ++n)
		{
			// Calculate dError_dValue by iterating the next layer
			double dError_dValue = 0;

			for (size_t nn = 0; nn < nextLayerSize; ++nn)
			{
				// Use the cached data from the previous iteration
				dError_dValue += (*next_dError_dNetValues)[nn] * oldWeights[sinapseIdx++];
			}

			double value = m_neurons[l][n].value;
			double dValue_dNetValue = activationFunctionDerivative(value);
			
			// Cache dError_dNetValue for the previous layer to use
			double dError_dNetValue = dError_dValue * dValue_dNetValue;
			(*dError_dNetValues)[n] = dError_dNetValue;

			sinapseIdx = 0;

			for (size_t p = 0; p < previousLayerSize; ++p)
			{
				Sinapse* sinapse = getSinapse(previousLayer, p, n);
				double dError_dWeight = m_neurons[previousLayer][p].value;
				
				// Store the old weights
				oldWeights[sinapseIdx++] = sinapse->weight;
				
				// Update weight
				sinapse->weight -= rate * dError_dNetValue * dError_dWeight;
			}
		}

		// Swap references
		Array<double>* tmp = next_dError_dNetValues;
		next_dError_dNetValues = dError_dNetValues;
		dError_dNetValues = tmp;
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
			std::cout << "(" << m_neurons[l][n].value << ", " << m_neurons[l][n].bias << ") ";
		}

		std::cout << std::endl;

		if ( l < numLayers - 1)
		{
			std::cout << " S" << l << ": ";
			for (int s = 0 ; s < m_sinapses[l].size(); ++s)
			{
				std::cout << "(" << m_sinapses[l][s].weight << ") ";
			}
		}

		std::cout << std::endl;
	}
}

