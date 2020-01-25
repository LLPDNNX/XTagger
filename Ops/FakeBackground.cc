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


REGISTER_OP("FakeBackground")
    .Input("batch: float32")
    .Input("is_signal: bool")
    .Output("fake_batch: float32")
    .Attr("feature_index: int")
    .Attr("rstart: float = -2")
    .Attr("rend: float = 5")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        c->set_output(0,c->input(0));
        return Status::OK();
    });

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include "TH1F.h"
#include "TFile.h"

#include <random>

namespace 
{
    static thread_local std::unique_ptr<std::mt19937> generator_;
}


class FakeBackgroundOp:
    public OpKernel
{

    private:
        int feature_index_;
        float rstart_;
        float rend_;
    public:
        explicit FakeBackgroundOp(OpKernelConstruction* context): 
            OpKernel(context)
        {
            OP_REQUIRES_OK(
                context,
                context->GetAttr("feature_index",&feature_index_)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("rstart",&rstart_)
            );
            OP_REQUIRES_OK(
                context,
                context->GetAttr("rend",&rend_)
            );
            if (rstart_>=rend_)
            {
                context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,
                    "Default range of parameter ["+
                    std::to_string(rstart_)+
                    "; "+
                    std::to_string(rend_)+
                    "] not viable. [a,b] with b>a required!" 
                ));
            }
        }
        
        virtual ~FakeBackgroundOp()
        { 
        }
        
        void Compute(OpKernelContext* context)
        {
            if (not generator_)
            {
                generator_.reset(new std::mt19937(123456));
            }
            const Tensor& batch_tensor = context->input(0);
            auto batch = batch_tensor.flat<float>();
            
            const Tensor& is_signal_tensor = context->input(1);
            auto is_signal = is_signal_tensor.flat<bool>();
            
            int64_t n_batches = batch_tensor.dim_size(0);
            int64_t batch_length = batch_tensor.NumElements()/n_batches;
            if (n_batches!=is_signal_tensor.dim_size(0))
            {
                context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,"Batch ("+
                    std::to_string(n_batches)+
                    ") and label tensor ("+
                    std::to_string(is_signal_tensor.dim_size(0))+
                    ") need to have the same first dimension"));
                return;
            }
            
            Tensor* fake_batch_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output("fake_batch", batch_tensor.shape(),&fake_batch_tensor));
            auto fake_batch = fake_batch_tensor->flat<float>();
            
            std::vector<float> signal_dist;
            for (int64_t ibatch = 0; ibatch < n_batches; ++ibatch)
            {
                if (is_signal(ibatch))
                {
                    signal_dist.push_back(batch(ibatch*batch_length+feature_index_));
                }
            }
            
            if (signal_dist.size()==0)
            {
                std::uniform_real_distribution<float> dist(rstart_,rend_);

                for (int64_t ibatch = 0; ibatch < n_batches; ++ibatch)
                {
                    for (int64_t ielem = 0; ielem < batch_length; ++ielem)
                    {
                        if (ielem == feature_index_)
                        {
                            fake_batch(ibatch*batch_length+ielem) = dist(*generator_);
                        }
                        else
                        {
                            fake_batch(ibatch*batch_length+ielem) = batch(ibatch*batch_length+ielem);
                        }
                    }
                }
            }
            else if (signal_dist.size()==1)
            {
                for (int64_t ibatch = 0; ibatch < n_batches; ++ibatch)
                {
                    for (int64_t ielem = 0; ielem < batch_length; ++ielem)
                    {
                        if (!is_signal(ibatch) and ielem == feature_index_)
                        {
                            fake_batch(ibatch*batch_length+ielem) = signal_dist[0];
                        }
                        else
                        {
                            fake_batch(ibatch*batch_length+ielem) = batch(ibatch*batch_length+ielem);
                        }
                    }
                }
            }
            else 
            {
                std::uniform_int_distribution<unsigned int> dist(0, static_cast<unsigned int>(signal_dist.size()-1));
                for (int64_t ibatch = 0; ibatch < n_batches; ++ibatch)
                {
                    for (int64_t ielem = 0; ielem < batch_length; ++ielem)
                    {
                        if (!is_signal(ibatch) and ielem == feature_index_)
                        {
                            const unsigned int v = dist(*generator_);

                            fake_batch(ibatch*batch_length+ielem) = signal_dist[v];
                        }
                        else
                        {
                            fake_batch(ibatch*batch_length+ielem) = batch(ibatch*batch_length+ielem);
                        }
                    }
                }
            }
        }
};

REGISTER_KERNEL_BUILDER(Name("FakeBackground").Device(DEVICE_CPU),FakeBackgroundOp)

