
#include "NeuralNet.h"

#include <math.h>
#include <stdlib.h>
#include <time.h>

NeuralNet::NeuralNet(Array<size_t>& neuronsPerLayer) : m_maxLayerSize(0), m_maxSinapsesInLayer(0)
{
	size_t numLayers = neuronsPerLayer.size();

	//srand(0);

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

void NeuralNet::backPropagate(Array<double>& inputValues,
	Array<double>& outputValues, double rate, double biasRate)
{
	//tODO confirm output and input sizes
	forwardPropagate(inputValues);

	// A neuron value model:
	//
	// previousValue -> weight -> net value -> value
	//
	// With net value being the pre-activation and value the post-activation.
	// We want to find the contribution of a weight on the total error:
	//
	// [1] dError/dWeight = dError/dValue * dValue/dNetValue * dNetValue/dWeight
	//
	// To update a weight (gradient descent):
	//
	// [2] weight -= rate * dError/dWeight
	//
	// On each back-propagation step the error needs to be propagated to the
	// previous layer. The values that we propagate (delta) are the contribution
	// of the values of each neuron on the previous layer to the total error
	// via the sinapses on each neuron of the current layer. These values are
	// per incoming sinapse of each current layer's neuron:
	//
	// [3] delta = dError/dPreviousNeuronValue
	//           = dError/dNetValue * dNetValue/dPreviousNeuronValue
	//           = dError/dNetValue * weight
	// 
	// We use two arrays for the deltas. We swap them back and forth (using
	// pointers to them) as the current array and the next layer's array. The
	// arrays need to have a size as large as the largest number of sinapses on
	// a layer.

	// Delare and init a bunch of variables once only.
	size_t outputLayer = m_neurons.size() - 1;
	double previousLayer = outputLayer - 1;
	double layerSize = getLayerSize(outputLayer);
	double previousLayerSize = getLayerSize(previousLayer);
	double nextLayerSize = 0;
	double value;
	double dError_dValue;
	double dValue_dNetValue;
	double dError_dNetValue;
	double dNetValue_dWeight;
	double dError_dWeight;
	Array<double>* tmpArray;

	// Iterate from the output layer back to the first hidden layer (not the
	// input - it has no weights to adjust)
	for (size_t l = outputLayer; l > 0; --l)
	{
		// Iterate throught the neurons of the layer
		for (size_t n = 0; n < layerSize; ++n)
		{
			// The neuron value
			value = m_neurons[l][n].value;

			// The contribution of the value to the Error
			if (l == outputLayer)
			{
				// Output layer: 
				// [4] dError/dValue = (value - expected)
				dError_dValue = value - outputValues[n];
			}
			else
			{
				// Hidden layer:
				// [5] dError/dValue = Sum(deltas from next layer)
				// see also equation 
				dError_dValue = 0;
				for (size_t nn = 0; nn < nextLayerSize; ++nn)
				{
					dError_dValue += getSinapse(l, n, nn)->delta;
				}
			}						

			// Calculate the dError/dNetValue:
			// [6] dError/dNetValue = dError/dValue * dValue/dNetValue
			dValue_dNetValue = activationFunctionDerivative(value);
			dError_dNetValue = dError_dValue * dValue_dNetValue;

			// Update the weights on each input
			for (size_t p = 0; p < previousLayerSize; ++p)
			{
				// Get the sinapse and weight
				Sinapse* sinapse = getSinapse(previousLayer, p, n);
				double weight = sinapse->weight;

				// Cache the delta (see equation [3])
				sinapse->delta = dError_dNetValue * weight;

				// Calculate dError/dWeight (see equaion [1]) and update the
				// weight (see equation [2])
				dNetValue_dWeight = m_neurons[previousLayer][p].value;
				dError_dWeight = dError_dNetValue * dNetValue_dWeight;
				sinapse->weight -= rate * dError_dWeight;
			}

			// Update the bias on the neuron
			// [7] dError/dBias = dError/dNetValue
			m_neurons[l][n].bias -= biasRate * dError_dNetValue;

		}

		// Update variables for the next iteration (except for the last
		// iteration, in which the layer is 1)
		if (l > 1)
		{
			previousLayer--;
			
			nextLayerSize = layerSize;
			layerSize = previousLayerSize;
			previousLayerSize = getLayerSize(previousLayer);
		}
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

