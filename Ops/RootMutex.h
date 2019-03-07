#ifndef ROOT_TF_ROOTMUTEX_H
#define ROOT_TF_ROOTMUTEX_H

#include "TError.h"

#include <mutex>

class RootMutex
{
    private:
        std::mutex _rmutex;
        
        RootMutex()
        {
            gErrorIgnoreLevel = 50000;
        }
        
    public:
        class Lock
        {
            friend RootMutex;
            private:
                std::mutex& _lockmutex;
                Lock(std::mutex& m):
                    _lockmutex(m)
                { 
                    _lockmutex.lock();  
                    //std::cout<<"lock aquired"<<std::endl;
                }
                
            public:
                Lock(const Lock& lock) = delete;
                Lock& operator=(const Lock& lock) = delete;
                Lock& operator=(Lock&& lock) = delete;
                Lock& operator=(Lock& lock) = delete;
                
                Lock(Lock&& lock):
                    _lockmutex(lock._lockmutex)
                {
                }
                
                ~Lock()
                {
                    //std::cout<<"lock released"<<std::endl;
                    _lockmutex.unlock();
                }
        };
        
        static Lock lock()
        {
            static RootMutex rootMutex;
            return rootMutex._rmutex;
        }  
};

#endif
