/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "NvInfer.h"
#include "NvOnnxConfig.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>

#include "common.h"
#include "buffers.h"

const int batchSize = 1;
const int skipSample = 1;
std::vector<std::string> inputTensorNames;
std::vector<std::string> outputTensorNames;
std::string serializePath("~/Documents/tensorrt-shog/tmp/shogi_resnet_serialize.bin");

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class SampleOnnxMNIST
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleOnnxMNIST()
        : mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Function deserialize the network engine from file
    //!
    bool load();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool SampleOnnxMNIST::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMIN, Dims4{1, 1, 28, 28});
    profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kOPT, Dims4{2, 1, 28, 28});
    profile->setDimensions(inputTensorNames[0].c_str(), OptProfileSelector::kMAX, Dims4{128, 1, 28, 28});
    config->addOptimizationProfile(profile);

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 2);
    IHostMemory *serializedModel = mEngine->serialize();

    ofstream serializedModelFile(serializePath, ios::binary);
    serializedModelFile.write((const char *)serializedModel->data(), serializedModel->size());

    return true;
}

bool SampleOnnxMNIST::load()
{
    ifstream serializedModelFile(serializePath, ios::in | ios::binary);
    serializedModelFile.seekg(0, ios_base::end);
    size_t fsize = serializedModelFile.tellg();
    serializedModelFile.seekg(0, ios_base::beg);
    std::vector<char> fdata(fsize);
    serializedModelFile.read((char *)fdata.data(), fsize);

    auto runtime = createInferRuntime(gLogger);
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(fdata.data(), fsize, nullptr), samplesCommon::InferDeleter());

    mInputDims = Dims4{1, 1, 28, 28};
    mOutputDims = Dims2{1, 10};

    return true;
}
//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleOnnxMNIST::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    // [W] [TRT] Calling isShapeTensor before the entire network is constructed may result in an inaccurate result.
    auto parsed = parser->parseFromFile(
        "data/mnist_cnn.onnx", static_cast<int>(gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(batchSize);//PARAM
    config->setMaxWorkspaceSize(16_MiB);
    if (false)//PARAM FP16
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (false)//PARAM FP16
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), -1);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool SampleOnnxMNIST::infer()
{
    gLogInfo << "createExecutionContext" << std::endl;
    auto context = mEngine->createExecutionContext();
    if (!context)
    {
        return false;
    }
    context->setBindingDimensions(0, Dims4{batchSize, 1, 28, 28});
    gLogInfo << "BufferManager" << std::endl;
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, batchSize, context);
    
    gLogInfo << "processInput" << std::endl;
    // Read the input data into the managed buffers
    assert(inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    gLogInfo << "copyInputToDevice" << std::endl;
    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    gLogInfo << "executeV2" << std::endl;
    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    gLogInfo << "copyOutputToHost" << std::endl;
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    gLogInfo << "verifyOutput" << std::endl;
    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool SampleOnnxMNIST::processInput(const samplesCommon::BufferManager& buffers)
{
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    ifstream fin("data/mnist_test_images.bin", ios::in|ios::binary);
    fin.seekg(inputH*inputW*sizeof(float)*skipSample);

    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(inputTensorNames[0]));
    fin.read((char*)hostDataBuffer, inputH*inputW*sizeof(float)*batchSize);

    return true;
}

bool compareResult(float *expected, float *actual, size_t size)
{
    float maxDiff = 0.0F;
    size_t maxDiffIdx = 0;
    for (size_t i = 0; i < size; i++)
    {
        float diff = abs(expected[i] - actual[i]);
        if (diff > maxDiff)
        {
            maxDiff = diff;
            maxDiffIdx = i;
        }
    }

    gLogInfo << "max diff among " << size << " elements: [" << maxDiffIdx << "] " << expected[maxDiffIdx] << "!=" << actual[maxDiffIdx] << std::endl;
    return maxDiff < 1e-3;
}
//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool SampleOnnxMNIST::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    const int outputSize = mOutputDims.d[1];
    float* output = static_cast<float*>(buffers.getHostBuffer(outputTensorNames[0]));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    gLogInfo << "Output:" << std::endl;
    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }

        gLogInfo << " Prob " << i << "  " << std::fixed << std::setw(5) << std::setprecision(4) << output[i] << " "
                 << "Class " << i << ": " << std::string(int(std::floor(output[i] * 10 + 0.5f)), '*') << std::endl;
    }
    gLogInfo << std::endl;

    return true;
}

bool checkSerializedFile()
{
    ifstream f(serializePath, ios::in | ios::binary);
    return f.is_open();
}

int main(int argc, char** argv)
{
    inputTensorNames.push_back("input_0");
    outputTensorNames.push_back("output_0");
    SampleOnnxMNIST sample;

    if (checkSerializedFile())
    {
        gLogInfo << "using serialized file" << std::endl;
        if (!sample.load())
        {
            gLogInfo << "load failed" << std::endl;
            return 1;
        }
    }
    else
    {
        if (!sample.build())
        {
            gLogInfo << "build failed" << std::endl;
            return 1;
        }
    }

    if (!sample.infer())
    {
        gLogInfo << "infer failed" << std::endl;
    }
}
