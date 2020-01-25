/*===================================================================
Copyright 2019 Matthias Komm, Vilius Cepaitis, Robert Bainbridge, 
Alex Tapper, Oliver Buchmueller. All Rights Reserved. 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
    
Unless required by applicable law or agreed to in writing, 
software distributed under the License is distributed on an "AS IS" 
BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express 
or implied.See the License for the specific language governing 
permissions and limitations under the License.
===================================================================*/


#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/framework/op.h"

#include <regex>

using namespace tensorflow;

REGISTER_OP("ClassificationWeights")
    .Input("labels: float32")
    .Input("input: float32")
    .Output("out: float32")
    .Attr("rootfile: string")
    .Attr("histnames: list(string)")
    .Attr("varindex: list(int)")
    .SetIsStateful()
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        tensorflow::shape_inference::ShapeHandle label_shape =  c->input(0);
        int64_t batch_dim = c->Value(c->Dim(label_shape,0));
        shape_inference::ShapeHandle s = c->MakeShape({batch_dim});
        c->set_output(0,s);
        return Status::OK();
    });

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "TH2.h"
#include "TFile.h"

#include "RootMutex.h"


class ClassificationWeightsOp:
    public OpKernel
{

    private:
        std::string filePath;
        std::vector<std::string> histNames;
        std::vector<std::shared_ptr<TH2>> hists;
        std::vector<int> varIndex;
        
    public:
        explicit ClassificationWeightsOp(OpKernelConstruction* context): 
            OpKernel(context)
        {
            OP_REQUIRES_OK(
                context,
                context->GetAttr("rootfile",&filePath)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("histnames",&histNames)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("varindex",&varIndex)
            );
            RootMutex::Lock lock = RootMutex::lock();
            TFile rootFile(filePath.c_str());
            if (not rootFile.IsOpen ())
            {
                context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,
                    "Root file '"+filePath+"' cannot be opened"
                ));
            }
            for (auto histName: histNames)
            {
                if (not rootFile.Get(histName.c_str()))
                {
                    context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,
                        "Cannot find hist '"+histName+"' in file '"+filePath+"'"
                    ));
                }
            
                TH2* hist = dynamic_cast<TH2*>(rootFile.Get(histName.c_str())->Clone());
                if (not hist)
                {
                    context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,
                        "Cannot cast hist '"+histName+"' to TH2 in file '"+filePath+"'"
                    ));
                }
                hist->SetDirectory(0);
                hists.push_back(std::shared_ptr<TH2>(hist));
            }
        }
        
        
        virtual ~ClassificationWeightsOp()
        { 
            RootMutex::Lock lock = RootMutex::lock();
            hists.clear();
        }
        
        float computeWeight(unsigned int classIndex, float value1, float value2)
        {
            TH2* hist = hists[classIndex].get();
            int bin = hist->FindBin(value1,value2);
            return static_cast<float>(hist->GetBinContent(bin));
        }

        void Compute(OpKernelContext* context)
        {
            const Tensor& label_tensor = context->input(0);
            auto label = label_tensor.flat<float>();
            int64_t num_batches = label_tensor.dim_size(0);
            int64_t label_length = label_tensor.dim_size(1);
            if (label_length<0 or static_cast<unsigned int>(label_length)!=hists.size())
            {
                context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,
                    "Labels ("+std::to_string(hists.size())+") need to be of same size as tensor ("+std::to_string(label_length)+")"
                ));
                return;
            }

            const Tensor& value_tensor = context->input(1);
            auto value = value_tensor.flat<float>();
            int64_t value_size = value_tensor.dim_size(1);

            Tensor* output_tensor = nullptr;
            TensorShape shape;
            shape.AddDim(num_batches);
            OP_REQUIRES_OK(context, context->allocate_output("out", shape,&output_tensor));
            auto output = output_tensor->flat<float>();

            for (unsigned int ibatch = 0; ibatch < num_batches; ++ibatch)
            {
                unsigned int class_index = 0;
                for (unsigned int i = 0; i < label_length; ++i)
                { 
                    if (label(ibatch*label_length+i)>0.5f)
                    {
                        class_index = i;
                        break;
                    }
                }
                float varValue1 = value(ibatch*value_size+varIndex[0]);
                float varValue2 = value(ibatch*value_size+varIndex[1]);
                output(ibatch) = computeWeight(class_index,varValue1,varValue2);
            }
        }  
};


REGISTER_KERNEL_BUILDER(Name("ClassificationWeights").Device(DEVICE_CPU),ClassificationWeightsOp)

