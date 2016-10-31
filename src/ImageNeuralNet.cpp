#include "ImageNeuralNet.h"

#include <OpenImageIO/imageio.h>

OIIO_NAMESPACE_USING


bool createImageNet(std::string imagePath, 
						  Array<size_t>& neuronsPerHiddenLayer,
						  size_t numOutputs,
						  NeuralNet* outNet)
{
	ImageInput *in = ImageInput::open (imagePath);
	if (!in)
		return false;
	
	const ImageSpec &spec = in->spec();
	int xres = spec.width;
	int yres = spec.height;
	int channels = spec.nchannels;
	int numPixels = xres * yres * channels;

	// Array with all layer sizes
	Array<size_t> neuronsPerLayer(neuronsPerHiddenLayer.size() + 2);

	// Input layer size
	neuronsPerLayer[0] = numPixels;

	// Hidden layer size	
	for (size_t i = 0; i < neuronsPerHiddenLayer.size(); ++i)
	{
		neuronsPerLayer[i + 1] = neuronsPerHiddenLayer[i];
	}

	// Output layer size
	neuronsPerLayer[neuronsPerHiddenLayer.size() + 1] = numOutputs;

	// Create NeuralNet
	outNet = new NeuralNet(neuronsPerLayer);

	in->close();
}

bool getPixels(std::string imagePath, Array<double>& pixels)
{
	// Open image and load its params
	ImageInput *in = ImageInput::open (imagePath);
	if (!in)
		return false;

	const ImageSpec &spec = in->spec();
	int xres = spec.width;
	int yres = spec.height;
	int channels = spec.nchannels;
	int numPixels = xres * yres * channels;

	// Read image
	Array<unsigned char> buffer(numPixels);
	in->read_image(TypeDesc::UINT8, buffer.getRef(0));

	in->close ();

	// Init the doubles array
	pixels.allocate(numPixels);
	for(int i = 0; i < pixels.size(); i++)
	{
		pixels[i] = ((double)buffer[i]) / 256.0;
	}

	return true;
}



