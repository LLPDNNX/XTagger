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
        for (unsigned int i = 1; i < c->num_inputs(); ++i)
        {
            tensorflow::shape_inference::ShapeHandle input_shape = c->input(i);
            tensorflow::shape_inference::ShapeHandle ouput_shape = input_shape;
            c->ReplaceDim(input_shape,0,c->MakeDim(-1),&ouput_shape);
            c->set_output(i-1,ouput_shape);
        }
        
        return Status::OK();
    })
    .Doc(R"doc(A Reader that outputs the lines of a file delimited by '\n'.)doc");

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
                //note: one cannot enforce order in which compute is called per batch
            }
            const Tensor& rate_tensor = context->input(0);
            auto rates = rate_tensor.flat<float>();
            
            std::vector<unsigned int> diced_rates(rates.size());
            unsigned int sum_rates = 0;
            for (unsigned int i = 0; i < rates.size(); ++i)
            {
                diced_rates[i]=std::poisson_distribution<unsigned int>(rates(i))(*generator_);
                sum_rates+=diced_rates[i];
            }
            for (unsigned int input_index = 1; input_index < context->num_inputs(); ++input_index)
            {
                DataType dataType = context->input_dtype(input_index);
                if (dataType==DT_FLOAT)
                {
                    ComputeTmpl<float>(context,input_index,diced_rates,sum_rates);
                }
                else if (dataType==DT_INT32)
                {
                    ComputeTmpl<int>(context,input_index,diced_rates,sum_rates);
                }
                else
                {
                    throw std::runtime_error("Data type '"+DataTypeString(dataType)+"' not (yet) supported");
                }
            }
        }
        
        template<class T> void ComputeTmpl(
            OpKernelContext* context, 
            unsigned int input_index, 
            const std::vector<unsigned int>& diced_rates,
            unsigned int sum_rates
        ) const
        {
            
            const Tensor& input_tensor = context->input(input_index);
            auto input_data = input_tensor.flat<T>();
            int batch_size = input_tensor.dim_size(0);
            int batch_length = input_tensor.NumElements()/batch_size;
            
            Tensor* output_tensor = nullptr;
            TensorShape output_shape = input_tensor.shape();
            output_shape.set_dim(0,sum_rates);
            OP_REQUIRES_OK(context, context->allocate_output(input_index-1, output_shape,&output_tensor));
            auto output_data = output_tensor->flat<T>();
            
            unsigned int output_batch_index = 0;
            for (unsigned int ibatch = 0; ibatch<batch_size; ++ibatch)
            {
                for (unsigned int icopy = 0; icopy<diced_rates[ibatch];++icopy)
                {
                    for (unsigned int idata = 0; idata<batch_length; ++idata)
                    {
                        output_data(output_batch_index*batch_length+idata) = input_data(ibatch*batch_length+idata);
                    }
                    output_batch_index+=1;
                }
            }
            if ( output_batch_index!=sum_rates) throw std::runtime_error("Output ("+std::to_string(output_batch_index)+") not equal sum ("+std::to_string(sum_rates)+")");
        }  
};

REGISTER_KERNEL_BUILDER(Name("Resampler").Device(DEVICE_CPU),ResamplerOp);

