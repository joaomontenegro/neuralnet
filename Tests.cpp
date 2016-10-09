
#include "NeuralNet.h"

#include <stdlib.h>
#include <iostream>

#define RUNTEST(F) if (!F) { failed++; }

#define EPSILON 0.00000001
#define DOUBLE_EQUALS(d0, d1) abs(d0 - d1) < EPSILON

bool testNeuralNetMethod(const char* funcName, int res, int value, int expected)
{
	bool pass = (res == expected);

	std::cout << "Testing " << funcName << "(" << value << "): " << res;
	if (pass)
	{
		std::cout << " | PASSED" << std::endl;
	}
	else
	{
		std::cout << " | NOT PASSED! (expected: " << expected << ")" << std::endl;
	}

	return pass;
}

bool testNeuralNetMethod(const char* funcName, int res, int value0, int value1, int expected)
{
	bool pass = (res == expected);

	std::cout << "Testing " << funcName << "(" << value0 << ", " << value1 << "): " << res;
	if (pass)
	{
		std::cout << " | PASSED" << std::endl;
	}
	else
	{
		std::cout << " | FAILED! (expected: " << expected << ")" << std::endl;
	}

	return pass;
}

int RunNeuralNetTests()
{
	size_t numInputs = 2;
	size_t numOutputs = 5;
	size_t numPerHiddenLayer = 3;
	size_t numHiddenLayers = 4;
	NeuralNet net(numInputs, numOutputs, numPerHiddenLayer, numHiddenLayers);
	
	int failed = 0;

	RUNTEST(testNeuralNetMethod("getLayer", net.getLayer(0), 0, 0));
	RUNTEST(testNeuralNetMethod("getLayer", net.getLayer(1), 1, 0));
	RUNTEST(testNeuralNetMethod("getLayer", net.getLayer(2), 2, 1));
	RUNTEST(testNeuralNetMethod("getLayer", net.getLayer(3), 3, 1));
	RUNTEST(testNeuralNetMethod("getLayer", net.getLayer(4), 4, 1));
	RUNTEST(testNeuralNetMethod("getLayer", net.getLayer(5), 5, 2));
	RUNTEST(testNeuralNetMethod("getLayer", net.getLayer(9), 9, 3));
	RUNTEST(testNeuralNetMethod("getLayer", net.getLayer(13), 13, 4));
	RUNTEST(testNeuralNetMethod("getLayer", net.getLayer(17), 17, 5));
	RUNTEST(testNeuralNetMethod("getLayer", net.getLayer(21), 21, -1));

	std::cout << std::endl;

	RUNTEST(testNeuralNetMethod("getFirstNeuronIndex", net.getFirstNeuronIndex(0), 0, 0));
	RUNTEST(testNeuralNetMethod("getFirstNeuronIndex", net.getFirstNeuronIndex(1), 1, 2));
	RUNTEST(testNeuralNetMethod("getFirstNeuronIndex", net.getFirstNeuronIndex(2), 2, 5));
	RUNTEST(testNeuralNetMethod("getFirstNeuronIndex", net.getFirstNeuronIndex(3), 3, 8));
	RUNTEST(testNeuralNetMethod("getFirstNeuronIndex", net.getFirstNeuronIndex(4), 4, 11));
	RUNTEST(testNeuralNetMethod("getFirstNeuronIndex", net.getFirstNeuronIndex(5), 5, 14));
	RUNTEST(testNeuralNetMethod("getFirstNeuronIndex", net.getFirstNeuronIndex(6), 6, -1));

	std::cout << std::endl;

	RUNTEST(testNeuralNetMethod("getNumNeurons", net.getNumNeurons(0), 0, 2));
	RUNTEST(testNeuralNetMethod("getNumNeurons", net.getNumNeurons(1), 1, 3));
	RUNTEST(testNeuralNetMethod("getNumNeurons", net.getNumNeurons(2), 2, 3));
	RUNTEST(testNeuralNetMethod("getNumNeurons", net.getNumNeurons(3), 3, 3));
	RUNTEST(testNeuralNetMethod("getNumNeurons", net.getNumNeurons(4), 4, 3));
	RUNTEST(testNeuralNetMethod("getNumNeurons", net.getNumNeurons(5), 5, 5));
	RUNTEST(testNeuralNetMethod("getNumNeurons", net.getNumNeurons(6), 6, 0));

	std::cout << std::endl;

	RUNTEST(testNeuralNetMethod("getOutNeuronIndex", net.getOutNeuronIndex(1, 0), 1, 0, 2));
	RUNTEST(testNeuralNetMethod("getOutNeuronIndex", net.getOutNeuronIndex(2, 1), 2, 1, 6));
	RUNTEST(testNeuralNetMethod("getOutNeuronIndex", net.getOutNeuronIndex(3, 1), 3, 1, 6));
	RUNTEST(testNeuralNetMethod("getOutNeuronIndex", net.getOutNeuronIndex(6, 4), 6, 4, -1));
	RUNTEST(testNeuralNetMethod("getOutNeuronIndex", net.getOutNeuronIndex(12, 4), 12, 4, 18));
	RUNTEST(testNeuralNetMethod("getOutNeuronIndex", net.getOutNeuronIndex(17, 0), 17, 0, -1));
	RUNTEST(testNeuralNetMethod("getOutNeuronIndex", net.getOutNeuronIndex(21, 0), 21, 0, -1));

	std::cout << std::endl;

	RUNTEST(testNeuralNetMethod("getInNeuronIndex", net.getInNeuronIndex(0, 0), 0, 0, -1));
	RUNTEST(testNeuralNetMethod("getInNeuronIndex", net.getInNeuronIndex(0, 1), 0, 1, -1));
	RUNTEST(testNeuralNetMethod("getInNeuronIndex", net.getInNeuronIndex(2, 0), 2, 0, 0));
	RUNTEST(testNeuralNetMethod("getInNeuronIndex", net.getInNeuronIndex(3, 0), 3, 0, 0));
	RUNTEST(testNeuralNetMethod("getInNeuronIndex", net.getInNeuronIndex(3, 2), 3, 2, -1));
	RUNTEST(testNeuralNetMethod("getInNeuronIndex", net.getInNeuronIndex(6, 1), 6, 1, 3));
	RUNTEST(testNeuralNetMethod("getInNeuronIndex", net.getInNeuronIndex(12, 0), 12, 0, 8));
	RUNTEST(testNeuralNetMethod("getInNeuronIndex", net.getInNeuronIndex(16, 2), 16, 2, 13));
	RUNTEST(testNeuralNetMethod("getInNeuronIndex", net.getInNeuronIndex(18, 4), 18, 4, -1));	

	std::cout << std::endl;

	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(0, 0), 0, 0, -1));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(1, 1), 1, 1, -1));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(2, 0), 2, 0, 0));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(2, 1), 2, 1, 3));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(4, 1), 4, 1, 5));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(5, 1), 5, 1, 9));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(7, 0), 7, 0, 8));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(7, 2), 7, 2, 14));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(7, 3), 7, 3, -1));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(9, 0), 9, 0, 16));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(9, 0), 9, 0, 16));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(11, 0), 11, 0, 24));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(11, 1), 11, 1, 27));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(13, 2), 13, 2, 32));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(14, 0), 14, 0, 33));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(16, 2), 16, 2, 45));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(17, 1), 17, 1, 41));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(17, 5), 17, 5, -1));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(18, 2), 18, 2, 47));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(18, 3), 18, 3, -1));
	RUNTEST(testNeuralNetMethod("getInSinapseIndex", net.getInSinapseIndex(21, 1), 21, 1, -1));
	
	std::cout << std::endl;

	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(0, 0), 0, 0, 0));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(1, 2), 1, 2, 5));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(3, 0), 3, 0, 9));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(3, 2), 3, 2, 11));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(3, 5), 3, 5, -1));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(5, 1), 5, 1, 16));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(7, 1), 7, 1, 22));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(9, 1), 9, 1, 28));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(9, 2), 9, 2, 29));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(11, 0), 11, 0, 33));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(11, 4), 11, 4, 37));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(13, 2), 13, 2, 45));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(11, 5), 11, 5, -1));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(16, 1), 16, 1, -1));
	RUNTEST(testNeuralNetMethod("getOutSinapseIndex", net.getOutSinapseIndex(21, 1), 21, 1, -1));
	
	std::cout << std::endl;

	return failed;
}

int RunForwardPropagateTests()
{
	size_t numInputs = 2;
	size_t numOutputs = 2;
	size_t numPerHiddenLayer = 2;
	size_t numHiddenLayers = 2;
	NeuralNet net(numInputs, numOutputs, numPerHiddenLayer, numHiddenLayers);

	Array<double> weights;
	weights.allocate(12);
	weights[0] = 0.1;
	weights[1] = 0.2;
	weights[2] = 0.3;
	weights[3] = 0.4;
	weights[4] = 0.5;
	weights[5] = 0.6;
	weights[6] = 0.7;
	weights[7] = 0.8;
	weights[8] = 0.9;
	weights[9] = 0.1;
	weights[10] = 0.2;
	weights[11] = 0.3;
	net.setWeights(weights);

	Array<double> inputs;
	Array<double> outputs;
	inputs.allocate(numInputs);
	outputs.allocate(numOutputs);

	inputs[0] = 0;
	inputs[1] = 1;

	net.forwardPropagate(inputs, outputs);

	std::cout << "Testing forwardPropagate() : " << outputs[0] << " , " << outputs[1];

	if (DOUBLE_EQUALS(outputs[0], 0.487) && DOUBLE_EQUALS(outputs[1], 0.193))
	{
		std::cout << " | PASSED" << std::endl << std::endl;
	}
	else
	{
		std::cout << " | FAILED (expected 0.487 , 0.193)" << std::endl << std::endl;
	}
	
	return 0;
}

int RunAllTests()
{
	int failed = 0;

	failed += RunNeuralNetTests();
	failed += RunForwardPropagateTests();

	if (failed > 0)
	{
		std::cout << std::endl << " *** " << failed << " FAILED TESTS *** " << std::endl;
	}
	else
	{
		std::cout << std::endl << "ALL TESTS PASSED..." << std::endl;
	}

	return failed;
}