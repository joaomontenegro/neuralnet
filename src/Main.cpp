#include "NeuralNet.h"
#include "Tests.h"

#include "math.h"
#include <iostream>
#include <sstream>


void train(int numRuns)
{
	Array<size_t> layerSizes(4);
	layerSizes[0] = 3;
	layerSizes[1] = 3;
	layerSizes[2] = 3;
	layerSizes[3] = 3;

	NeuralNet net(layerSizes);
	net.randomize();


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
		net.backPropagate(inputs[i % 8], outputs[i % 8], 0.5);
	}

	Array<double> results(3);

	for (unsigned int i = 0; i < 8; ++i)
	{
		net.forwardPropagate(inputs[i]);
		net.getOutputs(results);

		std::cout << round(inputs[i][0]) << " " << round(inputs[i][1]) << " " << round(inputs[i][2])
		          << " -> "
		          << round(results[0]) << " " << round(results[1]) << " " << round(results[2]) << " : ";

		std::cout.precision(5);
		
		std::cout << "ERROR: " << net.error(inputs[i], outputs[i])
		          << std::endl;
	}


}

int main(int argc, char* argv[])
{
	//RunAllTests();

	int numRuns = 1000;
	if (argc > 1)
	{
		std::stringstream convert(argv[1]);
		convert >> numRuns;
	}
	train(numRuns);

	return 0;
}
