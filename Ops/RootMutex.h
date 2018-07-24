#ifndef ROOT_TF_ROOTMUTEX_H
#define ROOT_TF_ROOTMUTEX_H

#include <mutex>

class RootMutex
{
    private:
        std::mutex _rmutex;
        RootMutex()
        {
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
