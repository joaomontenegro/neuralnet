#include "NeuralNet.h"
#include "TrainIncrement.h"

#include "math.h"
#include <iostream>
#include <sstream>


void train(int numRuns, float speedWeights, float speedBias)
{
	Array<size_t> layerSizes(3);
	layerSizes[0] = 3;
	layerSizes[1] = 6;
	layerSizes[2] = 3;

	NeuralNet net(layerSizes);
	net.randomize();

	std::cout << "  -----------------------   " << std::endl;
	net.print(false);
	std::cout << std::endl;

	typedef Array<double> DoubleArray;

	Array<DoubleArray> inputs(8);
	Array<DoubleArray> outputs(8);

	for (unsigned int i = 0; i < 8; ++i)
	{
		inputs[i].allocate(3);
		outputs[i].allocate(3);

		inputs[i][0] = (i & 4) >> 2;
		inputs[i][1] = (i & 2) >> 1;
		inputs[i][2] = (i & 1);

		outputs[i][0] = ((i + 1) & 4) >> 2;
		outputs[i][1] = ((i + 1) & 2) >> 1;
		outputs[i][2] = ((i + 1) & 1);
	}

	for (unsigned int i = 0; i < numRuns; ++i)
	{
		net.backPropagate(inputs[i % 8], outputs[i % 8], speedWeights, speedBias);
	}

	Array<double> results(3);

	double totalError = 0;

	for (unsigned int i = 0; i < 8; ++i)
	{
		net.forwardPropagate(inputs[i]);
		net.getOutputs(results);

		std::cout << round(inputs[i][0]) << " " << round(inputs[i][1]) << " " << round(inputs[i][2])
		          << " -> "
		          << round(results[0]) << " " << round(results[1]) << " " << round(results[2]) << " : ";

		std::cout << (((i + 1) & 4) >> 2 != round(results[0])? 'X' : '_')
		          << (((i + 1) & 2) >> 1 != round(results[1])? 'X' : '_')
		          << (((i + 1) & 1) >> 0 != round(results[2])? 'X' : '_') << " ";


		std::cout.precision(3);

		double error = net.error(inputs[i], outputs[i]);
		std::cout << error << "   | "
				  << results[0] << " " << results[1] << " " << results[2] << " : "
		          << std::endl;

		totalError += error;
	}

	std::cout << "AVG ERROR: " << totalError / 8 << std::endl;

	std::cout << std::endl;
	net.print(false);
	std::cout << "  -----------------------   " << std::endl;

}


void train2(int numRuns, float speedWeights, float speedBias)
{
	Array<size_t> layerSizes(3);
	layerSizes[0] = 2;
	layerSizes[1] = 2;
	layerSizes[2] = 2;

	NeuralNet net(layerSizes);
	net.randomize();

	typedef Array<double> DoubleArray;

	Array<DoubleArray> inputs(4);
	Array<DoubleArray> outputs(4);

	for (unsigned int i = 0; i < inputs.size(); ++i)
	{
		inputs[i].allocate(2);
		outputs[i].allocate(2);
	}

	//AND
	inputs[0][0] = 0; inputs[0][1] = 0; outputs[0][0] = 0;
	inputs[1][0] = 0; inputs[1][1] = 1; outputs[1][0] = 0;
	inputs[2][0] = 1; inputs[2][1] = 0; outputs[2][0] = 0;
	inputs[3][0] = 1; inputs[3][1] = 1; outputs[3][0] = 1;
	


	for (unsigned int i = 0; i < numRuns; ++i)
	{
		net.backPropagate(inputs[i % inputs.size()], outputs[i % outputs.size()], speedWeights, speedBias);
	}

	Array<double> results(outputs[0].size());

	for (unsigned int i = 0; i < inputs.size(); ++i)
	{
		net.forwardPropagate(inputs[i]);
		net.getOutputs(results);

		std::cout << round(inputs[i][0]) << " " << round(inputs[i][1])
		          << " -> "
		          << round(results[0]) << " : ";

		std::cout.precision(3);
		
		std::cout << "ERROR: " << net.error(inputs[i], outputs[i]) << "   | "
				  << results[0]
		          << std::endl;
	}
}

int trainIncrement(int argc, char* argv[])
{
	//RunAllTests();

	int numRuns = 1000;
	double speedWeights = 0.2;
	double speedBias = 0.2;

	if (argc > 1)
	{
		std::stringstream convertNumRuns(argv[1]);
		convertNumRuns >> numRuns;
	}

	if (argc > 2)
	{
		std::stringstream convertSpeedWeight(argv[2]);
		convertSpeedWeight >> speedWeights;
	}

	if (argc > 3)
	{
		std::stringstream convertSpeedBias(argv[3]);
		convertSpeedBias >> speedBias;

	}
	train(numRuns, speedWeights, speedBias);

	//train2(numRuns, speedWeights, speedBias);

	return 0;
}
