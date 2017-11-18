#include "sampleImageCLP.h"

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkComposeImageFilter.h>
#include <itkVectorImage.h>
#include <itkImageFileWriter.h>
#include <itkImageRandomNonRepeatingIteratorWithIndex.h>
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

typedef unsigned short VectorImagePixelType;
typedef itk::VectorImage<VectorImagePixelType, Dimension> VectorImageType;  
typedef VectorImageType::IndexType VectorImageIndexType;
typedef itk::ComposeImageFilter< InputImageType, VectorImageType> ComposeImageFilterType;
typedef itk::ImageRandomNonRepeatingConstIteratorWithIndex<VectorImageType> InputRandomIteratorType;
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


	InputImageType::Pointer maskimage = 0;

	if(maskImageFilename.compare("") != 0){
		InputImageFileReaderType::Pointer readimage = InputImageFileReaderType::New();
		readimage->SetFileName(maskImageFilename);
		readimage->Update();
		maskimage = readimage->GetOutput();	
	}


	VectorImageType::SizeType outputsize;
	outputsize[0] = neighborhood*2 + 1;
	outputsize[1] = neighborhood*2 + 1;
	outputsize[2] = neighborhood*2 + 1;

	char buf[50];	

	VectorImageType::RegionType region = vectorimage->GetLargestPossibleRegion();

	for(int i = 0; i < Dimension; i++){
		region.SetIndex(i, region.GetIndex(i) + neighborhood);
		region.SetSize(i, region.GetSize(i) - 2*neighborhood);
	}

	cout<<region<<endl;
	

	InputRandomIteratorType randomit(vectorimage, region);
	randomit.SetNumberOfSamples(vectorimage->GetLargestPossibleRegion().GetNumberOfPixels() - (pow(neighborhood,3)));
	randomit.GoToBegin();



	while(!randomit.IsAtEnd() && numSamples != 0){

		VectorImageIndexType index = randomit.GetIndex();
		bool extractsample = true;

		VectorImageIndexType n_index;
		n_index[0] = index[0] - neighborhood;
		n_index[1] = index[1] - neighborhood;
		n_index[2] = index[2] - neighborhood;

		if(maskimage){
			extractsample = maskimage->GetPixel(index) != 0;
		}

		if(extractsample){
			VectorImageType::RegionType outputRegion;
			outputRegion.SetIndex(n_index);
			outputRegion.SetSize(outputsize);		

			ExtractImageFilterType::Pointer extractimage = ExtractImageFilterType::New();
			extractimage->SetInput(vectorimage);			
			extractimage->SetExtractionRegion(outputRegion);
			extractimage->Update();

			string outfilename;

			if(prefix.compare("")!=0){
				sprintf(buf, "%d-%d-%d", n_index[0], n_index[1], n_index[2]);
				outfilename = outputImageDirectory + "/" + prefix + string(buf) + ".nrrd";
			}else{
				char *uuid = new char[100];	
				uuid_t id;
				uuid_generate(id);
			  	uuid_unparse(id, uuid);
			  	outfilename = outputImageDirectory + "/" + string(uuid) + ".nrrd";
			}

			

			cout<<"Writing: "<<outfilename<<endl;
			writer->SetFileName(outfilename);
			writer->SetInput(extractimage->GetOutput());
			writer->Update();
			numSamples--;
		}

		++randomit;
	}


	return EXIT_SUCCESS;
}