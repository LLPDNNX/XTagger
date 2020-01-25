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
