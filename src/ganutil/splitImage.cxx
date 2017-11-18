#include "splitImageCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkComposeImageFilter.h>
#include <itkVectorImage.h>
#include <itkImageFileWriter.h>
#include <itkNeighborhoodIterator.h>
#include <itkFixedArray.h>
#include <itkExtractImageFilter.h>

#include <uuid/uuid.h>
#include <iostream>

using namespace std;

typedef unsigned short PixelType;
static const int Dimension = 3;

typedef itk::Image<PixelType, Dimension> InputImageType;
typedef InputImageType::IndexType InputImageIndexType;
typedef itk::ImageFileReader<InputImageType> InputImageFileReaderType;
typedef itk::NeighborhoodIterator<InputImageType> InputIteratorType;

typedef unsigned short VectorImagePixelType;
typedef itk::VectorImage<VectorImagePixelType, Dimension> VectorImageType;  
typedef VectorImageType::IndexType VectorImageIndexType;
typedef itk::ComposeImageFilter< InputImageType, VectorImageType> ComposeImageFilterType;
typedef itk::NeighborhoodIterator<VectorImageType> VectorImageIteratorType;
typedef VectorImageIteratorType::RadiusType VectorImageRadiusType;
typedef itk::ImageFileWriter<VectorImageType> VectorImageFileWriterType;

typedef itk::ExtractImageFilter< VectorImageType, VectorImageType > ExtractImageFilterType;

int main (int argc, char * argv[]){


	PARSE_ARGS;
	
	if(vectorImageFilename.size() == 0){
		cout<<"Type "<<argv[0]<<" --help to find out how to use this program."<<endl;
		return 1;
	}

	if(outputImageDirectory.compare("") != 0){
		outputImageDirectory.append("/");
	}

	ComposeImageFilterType::Pointer composeImageFilter = ComposeImageFilterType::New();

	for(int i = 0; i < vectorImageFilename.size(); i++){
		cout<<"Reading:"<<vectorImageFilename[i]<<endl;
		InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
		readimage->SetFileName(vectorImageFilename[i]);
		readimage->Update();

		composeImageFilter->SetInput(i, readimage->GetOutput());
	}

	composeImageFilter->Update();

	VectorImageType::Pointer vectorimage = composeImageFilter->GetOutput();
	
	VectorImageType::SizeType size = vectorimage->GetLargestPossibleRegion().GetSize();
	VectorImageFileWriterType::Pointer writer = VectorImageFileWriterType::New();


	VectorImageType::SizeType outputsize;
	outputsize[0] = neighborhood*2;
	outputsize[1] = neighborhood*2;
	outputsize[2] = neighborhood*2;

	char buf[50];

	int stepi = floor(size[0]/round(size[0]/(neighborhood*2.0)));
	int stepj = floor(size[1]/round(size[1]/(neighborhood*2.0)));
	int stepk = floor(size[2]/round(size[2]/(neighborhood*2.0)));

	if(stepi == 0 || stepj == 0 || stepk == 0 ){
		cerr<<"Neighborhood larger than input image. Aborting cropping."<<endl;
		return EXIT_FAILURE;
	}

	for(int i = 0; i <= size[0] - outputsize[0]; i+=stepi){
		for(int j = 0; j <= size[1] - outputsize[1]; j+=stepj){
			for(int k = 0; k <= size[2] - outputsize[2]; k+=stepk){
				VectorImageIndexType index;
				index[0] = i;
				index[1] = j;
				index[2] = k;

				VectorImageType::RegionType outputRegion;
				outputRegion.SetIndex(index);
				outputRegion.SetSize(outputsize);
				

				ExtractImageFilterType::Pointer extractimage = ExtractImageFilterType::New();
				extractimage->SetInput(vectorimage);
				extractimage->SetDirectionCollapseToIdentity();
				extractimage->SetExtractionRegion(outputRegion);
				extractimage->Update();

				sprintf(buf, "%d-%d-%d", i, j, k);
				string outfilename = outputImageDirectory + "/" + prefix + string(buf) + ".nrrd";

				cout<<"Writing: "<<outfilename<<endl;
				writer->SetFileName(outfilename);
				writer->SetInput(extractimage->GetOutput());
				writer->Update();

			}
		}
	}


	return EXIT_SUCCESS;
}