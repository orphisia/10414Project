Located inside of the GPU's [[Streaming Multiprocessor]], fast but very small 

Block-level on chip SRAMs 
* Shared memory, software managed scratchpad shared among *all threads*
* Reduce number of trips to main memory 
	* threads in a block to stage memory 
	* stage data loaded in from VRAM 

[[Bank]] Conflicts - Shared memory is divided into 32 equally sized modules. If two threads make requests to different addresses in the same bank 
* Hardware serializes the accesses (bank conflict) which reduces bandwidth 
* If threads access distinct banks simultaneously (broadcast) the accesses complete simultaneously 



