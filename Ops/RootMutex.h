#ifndef ROOT_TF_ROOTMUTEX_H
#define ROOT_TF_ROOTMUTEX_H

#include <mutex>

class RootMutex
{
    private:
        std::mutex mutex;
        RootMutex()
        {
        }
        
    public:
        class Lock
        {
            friend RootMutex;
            private:
                std::mutex& _mutex;
                Lock(std::mutex& mutex):
                    _mutex(mutex)
                { 
                    _mutex.lock();  
                }
                
            public:
                ~Lock()
                {
                    _mutex.unlock();
                }
        };
        
        static Lock lock()
        {
            static RootMutex rootMutex;
            return rootMutex.mutex;
        }  
};

#endif
