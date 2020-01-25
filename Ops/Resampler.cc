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


REGISTER_OP("Resampler")
    .Attr("dtypes: list({int32, float})")
    .Input("rate: float")
    .Input("input: dtypes")
    .Output("output: dtypes")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) 
    {
        for (int i = 1; i < c->num_inputs(); ++i)
        {
            tensorflow::shape_inference::ShapeHandle input_shape = c->input(i);
            tensorflow::shape_inference::ShapeHandle ouput_shape = input_shape;
            TF_RETURN_IF_ERROR(c->ReplaceDim(input_shape,0,c->MakeDim(-1),&ouput_shape));
            c->set_output(i-1,ouput_shape);
        }
        
        return Status::OK();
    });

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <random>

namespace 
{
    static thread_local std::unique_ptr<std::mt19937> generator_;
}

class ResamplerOp:
    public OpKernel
{

    private:
        
    public:
        explicit ResamplerOp(OpKernelConstruction* context): 
            OpKernel(context)
        {
            
        }

        virtual ~ResamplerOp()
        { 
        }
        

        void Compute(OpKernelContext* context)
        {
            if (not generator_)
            {
                generator_.reset(new std::mt19937(123456));
            }
            const Tensor& rate_tensor = context->input(0);
            auto rates = rate_tensor.flat<float>();
            
            std::vector<unsigned int> diced_rates(static_cast<unsigned int>(rates.size()));
            unsigned int sum_rates = 0;
            for (unsigned int i = 0; i < rates.size(); ++i)
            {
                diced_rates[i]=std::poisson_distribution<unsigned int>(rates(i))(
                    *generator_
                );
                sum_rates+=diced_rates[i];
            }
            for (int input_index = 1; input_index < context->num_inputs(); ++input_index)
            {
                DataType dataType = context->input_dtype(input_index);
                if (dataType==DT_FLOAT)
                {
                    ComputeTmpl<float>(context,input_index,diced_rates,sum_rates);
                    if (not context->status().ok()) return;
                }
                else if (dataType==DT_INT32)
                {
                    ComputeTmpl<int>(context,input_index,diced_rates,sum_rates);
                    if (not context->status().ok()) return;
                }
                else
                {
                    context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,
                        "Data type '"+DataTypeString(dataType)+"' not (yet) supported"
                    ));
                    return;
                }
            }
        }
        
        template<class T> void ComputeTmpl(
            OpKernelContext* context, 
            int input_index, 
            const std::vector<unsigned int>& diced_rates,
            unsigned int sum_rates
        ) const
        {
            
            const Tensor& input_tensor = context->input(input_index);
            auto input_data = input_tensor.flat<T>();
            unsigned int batch_size = static_cast<unsigned int>(
                input_tensor.dim_size(0)
            );
            unsigned int batch_length = static_cast<unsigned int>(
                input_tensor.NumElements()/batch_size
            );
            
            Tensor* output_tensor = nullptr;
            TensorShape output_shape = input_tensor.shape();
            output_shape.set_dim(0,sum_rates);
            OP_REQUIRES_OK(context, context->allocate_output(
                input_index-1, output_shape,&output_tensor
            ));
            auto output_data = output_tensor->flat<T>();
            
            unsigned int output_batch_index = 0;
            for (unsigned int ibatch = 0; ibatch<batch_size; ++ibatch)
            {
                for (unsigned int icopy = 0; icopy<diced_rates[ibatch];++icopy)
                {
                    for (unsigned int idata = 0; idata<batch_length; ++idata)
                    {
                        output_data(output_batch_index*batch_length+idata) = \
                            input_data(ibatch*batch_length+idata);
                    }
                    output_batch_index+=1;
                }
            }
            if ( output_batch_index!=sum_rates) 
            {
                context->CtxFailureWithWarning(__FILE__, __LINE__, Status(error::INVALID_ARGUMENT,
                    "Output ("+std::to_string(output_batch_index)+
                    ") not equal sum ("+std::to_string(sum_rates)+")"
                ));
                return;
            }
        }  
};

REGISTER_KERNEL_BUILDER(Name("Resampler").Device(DEVICE_CPU),ResamplerOp)

