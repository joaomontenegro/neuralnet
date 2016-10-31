#ifndef _MNIST_H_
#define _MNIST_H_

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <fstream>

#include <Array.h>

#ifndef _SWAP4
#define _SWAP4(x) ((x >> 24) & 0xff) | ((x >> 8) & 0xff00) | ((x << 8) & 0xff0000)  | ((x << 24) & 0xff000000)
#endif


class MNIST
{
public:
	MNIST() : numImages(0), width(0), height(0), NUM_LABELS(10), net(0x0) {}
	~MNIST() {}

	typedef Array<uint8_t> PixelArray;
	typedef Array<PixelArray> ImageSet;
	Array<ImageSet> images;
	uint32_t numImages;
	uint32_t width;
	uint32_t height;

	uint32_t NUM_LABELS;

	NeuralNet* net;

	void print()
	{
		for (size_t label = 0; label < images.size(); ++label)
		{
			std::cout << " ------- LABEL " << label << " -------" << std::endl;
			for (size_t image = 0; image < images[label].size(); ++image)
			{
				printImage(label, image);

				std::cout << std::endl;
			}
		}
	}

	void printImage(size_t label, size_t imageN)
	{
		std::cout << " Image " << label << " : " << imageN << ":" << std::endl;

		size_t pixel = 0;
		for (size_t y = 0; y < height; ++y)
		{
			for (size_t x = 0; x < width; ++x)
			{
				if (images[label][imageN][y * width + x] != 0)
				{
					std::cout << '*';
				}
				else
				{
					std::cout << ' ';
				}
				
			}
			std::cout << std::endl;
		}
	}


	NeuralNet* train(std::string labelsPath, std::string imagesPath,
		                double rate, double biasRate,
		                int numSets=-1, int numTrainings=1)
	{
		if (!readTestImages(labelsPath, imagesPath))
		{
			return 0x0;
		}

		size_t maxImages = 0;
		for (size_t l = 0; l < NUM_LABELS; ++l)
		{
			if (images[l].size() > maxImages)
			{
				maxImages = images[l].size();
			}
		}

		Array<size_t> layerSizes(3);
		layerSizes[0] = width * height;
		layerSizes[1] = 50;
		layerSizes[2] = NUM_LABELS;

		if (net != 0x0)
		{
			delete net;
		}
		net = new NeuralNet(layerSizes);

		net->randomize();

		if (numSets < 0) numSets = numImages;

		int total = numSets * numTrainings;
		int counter = 0;
		int lastPercentage = -1;

		for (int t = 0; t < numTrainings; ++t)
		{
			std::cout << "Training Run " << t << std::endl;

			int n = numSets;

			for (size_t i = 0; i < maxImages; ++i)
			{
				for (size_t l = 0 ; l < NUM_LABELS; ++l)
				{

					int percentage = ++counter * 1000 / total;
					if (percentage > lastPercentage)
					{
						std::cout << "Training: "  << counter * 100.0 / total << "%" << std::endl;
						lastPercentage = percentage;
					}

					teach(l, i, rate, biasRate);

					if (--n == 0)  break;
					
				}

				if (n == 0)
				{
					break;
				}
			}

		}

		return net;
	}

	NeuralNet* load(std::string filepath)
	{
		if (net != 0x0)
		{
			delete net;
		}

		net = new NeuralNet();
		net->load(filepath.c_str());
	}

	uint32_t detect(PixelArray& image)
	{
		Array<double> inputs(width * height);
		Array<double> outputs;

		// Inputs : convert pixels to doubles
		for (size_t i = 0; i < inputs.size(); ++i)
		{
			inputs[i] = (double)image[i] / 256;
		}

		net->forwardPropagate(inputs);

		outputs.allocate(NUM_LABELS);
		net->getOutputs(outputs);

		uint32_t detectedLabel = 0;
		double maxSoFar = 0;
		for (size_t o = 0; o < NUM_LABELS; ++o)
		{
			if (outputs[o] > maxSoFar)
			{
				maxSoFar = outputs[o];
				detectedLabel = o;
			}

		}

		return detectedLabel;
	}

	double runTestSet(std::string labelsPath, std::string imagesPath)
	{
		readTestImages(labelsPath, imagesPath);

		int totalDetected = 0;

		for (size_t l = 0; l < NUM_LABELS; l++)
		{
			for (size_t i = 0; i < images[l].size(); i++)
			{
				uint32_t detected = detect(images[l][i]);

				if (detected == l)
				{
					totalDetected++;
					std::cout << "Yes:  ";

				}
				else
				{
					std::cout << "No :  ";
				}

				std::cout << l << " / " << detected << std::endl;

			}
		}

		return 1.0 - ((double)totalDetected) / numImages;
	}


private:
	bool readTestLabels(std::string filepath, Array<uint8_t>& labels)
	{
		std::cout << "Reading Test Labels: " << filepath << std::endl;

		// Open file
		std::ifstream ifs(filepath.c_str(), std::ios::in|std::ios::binary);
		if (!ifs.is_open())
	    {
	    	std::cerr << " *** MNIST: Could not open " << filepath
	    	          << "!" << std::endl;
	    	return false;
	    }

	    // Read magic number
	    uint32_t magicNumber;
	    ifs.read((char*)&magicNumber, 4);
	    magicNumber = _SWAP4(magicNumber);
	    if (magicNumber != 2049)
	    {
	    	std::cerr << " *** MNIST: Bad Magic Number for " << filepath
	    		      << " : " << magicNumber << std::endl;
	        ifs.close();
	    	return false;
	    }

	    // Read number of items
	    uint32_t numItems;
	    ifs.read((char*)&numItems, 4);
	    numItems = _SWAP4(numItems);
	    if (numItems <= 0)
	    {
	    	std::cerr << " *** MNIST: Bad number of items" << filepath
	    	          << " : " << numItems << std::endl;
	    	ifs.close();
	    	return false;
	    }

	    // Allocate array
	    labels.allocate(numItems);
	    ifs.read((char*)labels.getRef(0), numItems);

	    ifs.close();

	    return true;
	}

	bool readTestImages(std::string labelsFilepath,
					    std::string imagesFilepath)
	{
		// Read the labels
		Array<uint8_t> labels;
		if (!readTestLabels(labelsFilepath, labels))
		{
			return false;
		}

		std::cout << "Reading Test Images: " << imagesFilepath << std::endl;

		// Open file
		std::ifstream ifs(imagesFilepath.c_str(), std::ios::in|std::ios::binary);
		if (!ifs.is_open())
	    {
	    	std::cerr << " *** MNIST: Could not open " << imagesFilepath
	    	          << "!" << std::endl;
	    	return false;
	    }

	    // Read magic number
	    uint32_t magicNumber;
	    ifs.read((char*)&magicNumber, 4);
	    magicNumber = _SWAP4(magicNumber);
	    if (magicNumber != 2051)
	    {
	    	std::cerr << " *** MNIST: Bad Magic Number for " << imagesFilepath
	    		      << " : " << magicNumber << std::endl;
	    	ifs.close();
	    	return false;
	    }

	    // Read number of items
	    uint32_t numItems;
	    ifs.read((char*)&numItems, 4);
	    numItems = _SWAP4(numItems);
	    if (numItems != labels.size())
	    {
	    	std::cerr << " *** MNIST: Bad number of items" << imagesFilepath
	    	          << " : " << numItems << " doesn't match number of labels:"
	    	          << labels.size() << std::endl;
	    	ifs.close();
	    	return false;
	    }

	    numImages = numItems;

	    // Read width / height
	    ifs.read((char*)&height, 4);
	    ifs.read((char*)&width, 4);
	    height = _SWAP4(height);
	    width = _SWAP4(width);
	    if (height == 0 || width == 0)
	    {
	    	std::cerr << " *** MNIST: Bad width/height" << imagesFilepath
	    	          << " : " << width << "/" << height << std::endl;
	    	ifs.close();
	    	return false;
	    }

	    // Count the occurrences of each label
	    Array<size_t> occurrences(NUM_LABELS);
	    memset(occurrences.getRef(0), 0, NUM_LABELS * sizeof(size_t));
	    for (size_t i = 0; i < labels.size(); ++i)
	    {
	    	occurrences[labels[i]]++;
	    }

	    // Allocate the images array
	    images.allocate(NUM_LABELS);
	    for (size_t i = 0; i < NUM_LABELS; ++i)
	    {
	    	images[i].allocate(occurrences[i]);

	    	// Reset occurrences to be used now as a counter
	    	occurrences[i] = 0;
	    }

	    // Read the images
	    size_t imageSize = width * height;
	    for (size_t i = 0; i < numItems; ++i)
	    {
	    	size_t l = labels[i];
	    	size_t o = occurrences[l]++;

	    	images[l][o].allocate(imageSize);
	    	ifs.read((char*) images[l][o].getRef(0), imageSize);

	    }

	    ifs.close();

	    return true;
	}

	void teach(size_t label, size_t imageN, double rate, double biasRate)
	{
		if (images[label].size() <= imageN)
		{
			return;
		}

		// Expected value: all zero, except label set to 1
		Array<double> expected(NUM_LABELS);
		memset(expected.getRef(0), 0, NUM_LABELS * sizeof(double));
		expected[label] = 1.0f;


		// Inputs : convert pixels to doubles
		Array<double> inputs(width * height);
		for (size_t i = 0; i < inputs.size(); ++i)
		{
			inputs[i] = (double)images[label][imageN][i] / 256;
		}

		net->backPropagate(inputs, expected, rate, biasRate);
	}


};


#endif