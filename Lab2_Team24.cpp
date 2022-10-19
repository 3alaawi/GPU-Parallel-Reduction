// Please refer to the report for references to our code
#include <CL/sycl.hpp>
#include <array>
#include <iostream>
#include <queue>


class reduction_kernel;
namespace sycl = cl::sycl;
using namespace std;


int main(int, char**) {
   //here we initialized the array
    array<int32_t, 8> arr = { 1,2,3,4,5,6,7,8 }; 

    // we printed to the user the array that was inputed
    cout << "Contents of the array: "; 
    for (auto i : arr) {
        cout << i << " ";
    }
    cout << endl;

   // initialized the buffer
    sycl::buffer<int32_t, 1> arraybuffer(arr.data(), sycl::range<1>(arr.size())); 
   // initialized the queue
    sycl::queue Q;  


   // our workgroupsize is set to 32 because that works for most devices
    size_t workgroupsize = 32;
   //number of elements in the array that work-group reduces into
    auto reduced = workgroupsize * 2;
   // finding the number of work groups
    auto noofworkgroups = (arr.size() + reduced - 1) / reduced;

    Q.submit([&](sycl::handler& r) {
   // initializing the accessor
        sycl::accessor <int32_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local(sycl::range<1>(workgroupsize), r);
        auto global = arraybuffer.get_access<sycl::access::mode::read_write>(r);
   // parallel for loop
   /*
   we take the array from global to local where we perform all the calculations
   and then we resend the result from local to global and print out the result
   */
        r.parallel_for<>(
            sycl::nd_range<1>(workgroupsize, workgroupsize), [=](sycl::nd_item<1> item) {

               //initializing local and global ids
                size_t lid = item.get_local_linear_id();
                size_t gid = item.get_global_linear_id();
                local[lid] = 0;


                if ((2 * gid) < arr.size()) {
                    local[lid] = global[2 * gid] + global[2 * gid + 1];
                }

                item.barrier(sycl::access::fence_space::local_space);


                for (size_t i = 1; i < workgroupsize; i *= 2) {

                    auto j = 2 * i * lid;

                    if (j < workgroupsize) {
                        local[j] = local[j] + local[j + i];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (lid == 0) {
                    global[item.get_group_linear_id()] = local[0];
                }

            });
        });



    auto arrayaccessor = arraybuffer.get_access<sycl::access::mode::read>();
    cout << "Sum: "
        << arrayaccessor[0] << endl;
    return 0;
}