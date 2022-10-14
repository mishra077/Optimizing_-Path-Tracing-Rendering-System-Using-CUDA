// Importing necessary header files
#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"
#include <stdlib.h>
#include "moving_sphere.h"
#include "box.h"
#include <fstream>


// Error Handling in CUDA
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}


__device__ vec3 color(const ray& r, hitable **world, curandState *local_rand_state) {
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	vec3 cur_color = vec3(0.0, 0.0, 0.0);
	for (int i = 0; i < 50; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
			ray scattered;
			vec3 attenuation;
			vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_attenuation = emitted + cur_attenuation;
				cur_ray = scattered;
			}
			else {
				//cur_attenuation *= attenuation;
				//cur_attenuation = emitted + cur_attenuation;
				return cur_attenuation;
			}
		}
		else {
			return vec3(0.0, 0.0, 0.0);
			return cur_attenuation * vec3(0.005, 0.005, 0.005);
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f*(unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t)*vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
			return cur_attenuation * c;
		}
	}
	return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curand_init(1984, 0, 0, rand_state);
	}
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < ns; s++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
		ray r = (*cam)->get_ray(u, v, &local_rand_state);
		col += color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	col /= float(ns);
	float f = 0.3; //Gamma correction
	col[0] = 1.85 * sqrt(col[0]) / f;
	col[1] = 1.85 * sqrt(col[1]) / f;
	col[2] = 1.85 * sqrt(col[2]) / f;
	if (col[0] > 1) {
		col[0] = 1;
	}
	if (col[1] > 1) {
		col[1] = 1;
	}
	if (col[2] > 1) {
		col[2] = 1;
	}

	fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))


// DEFINING DIFFERENT SCENES // SCENE 0,1,2

// EQUAL WORKLOAD
// SCENE DESCRIPTION: A room with 2 entities, one cuboid and other cube
__device__ void scene_0(hitable **HIT_LIST, hitable **HIT_WORLD, curandState *rand_state) {
	int i = 0;
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *light = new diffuse_light_val(new constant_texture(vec3(15, 15, 15)));
	HIT_LIST[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));
	HIT_LIST[i++] = new yz_rect(0, 555, 0, 555, 0, red);
	HIT_LIST[i++] = new xz_rect(213, 343, 227, 332, 554, light);
	HIT_LIST[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
	HIT_LIST[i++] = new xz_rect(0, 555, 0, 555, 0, white);
	HIT_LIST[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));
	HIT_LIST[i++] = new translate(
		new rotate_y(new box(vec3(0, 0, 0), vec3(165, 165, 165), white), -18),
		vec3(130, 0, 65)
	);
	HIT_LIST[i++] = new translate(
		new rotate_y(new box(vec3(0, 0, 0), vec3(165, 330, 165), white), 15),
		vec3(265, 0, 295)
	);
	*HIT_WORLD = new hitable_list(HIT_LIST, i);
}

// UNEQUAL WORKLOAD
// SCENE DESCIPTION: THREE SPHERES
__device__ void scene_1(hitable **HIT_LIST, hitable **HIT_WORLD, curandState *rand_state) {
	texture_val *checker = new checker_texture(
		new constant_texture(vec3(0.2, 0.3, 0.1)),
		new constant_texture(vec3(0.9, 0.9, 0.9))
	);
	curandState local_rand_state = *rand_state;
	*rand_state = local_rand_state;
	HIT_LIST[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(checker));
	int i = 1;
	
	HIT_LIST[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
	HIT_LIST[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(new constant_texture(vec3(0.4, 0.2, 0.1))));
	HIT_LIST[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
	material *light = new diffuse_light_val(new constant_texture(vec3(15, 15, 15)));
	HIT_LIST[i++] = new xz_rect(213, 343, 227, 332, 554, light);
	*HIT_WORLD = new hitable_list(HIT_LIST, i);
}

// EXTREMELY UNEQUAL WORKLOAD
// SCENE DESCRIPTION: MULTIPLE SPHERES
__device__ void scene_2(hitable **HIT_LIST, hitable **HIT_WORLD, curandState *rand_state) {
	texture_val *checker = new checker_texture(
		new constant_texture(vec3(0.2, 0.3, 0.1)),
		new constant_texture(vec3(0.9, 0.9, 0.9))
	);
	curandState local_rand_state = *rand_state;
	*rand_state = local_rand_state;
	HIT_LIST[0] = new sphere(vec3(0, -1000, 0), 1000, new lambertian(checker));
	int i = 1;
	for (int a = -10; a < 10; a++) {
		for (int b = -10; b < 10; b++) {
			float choose_mat = RND;
			vec3 center(a + 0.9*RND, 0.2, b + 0.9*RND);
			if ((center - vec3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8f) {
					HIT_LIST[i++] = new moving_sphere(
						center,
						center + vec3(0, 0.5*RND, 0),
						0.0, 1.0, 0.2,
						new lambertian(new constant_texture(
							vec3(RND*RND,
								RND*RND,
								RND*RND))));
				}
				else if (choose_mat < 0.95f) {
					HIT_LIST[i++] = new sphere(center, 0.2,
						new metal(vec3(0.5f*(1.0f + RND), 0.5f*(1.0f + RND), 0.5f*(1.0f + RND)), 0.5f*RND));
				}
				else {
					HIT_LIST[i++] = new sphere(center, 0.2, new dielectric(1.5));
				}
			}
		}
	}
	HIT_LIST[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
	HIT_LIST[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(new constant_texture(vec3(0.4, 0.2, 0.1))));
	HIT_LIST[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
	material *light = new diffuse_light_val(new constant_texture(vec3(15, 15, 15)));
	HIT_LIST[i++] = new xz_rect(213, 343, 227, 332, 554, light);
	*HIT_WORLD = new hitable_list(HIT_LIST, i);
}



// CREATING THE WORLD 
// -> INTIALIZING THE SCENE TO SET UP
// -> PLACING THE CAMERA

// -> PLEASE COMMENT THE SCENES WHICH ARE NOT BEING RENDERED AND ALSO THE CAMERA PLACEMENT CODE FOR THE CORRESPOINING SCENE

__global__ void create_world(hitable **HIT_LIST, hitable **HIT_WORLD, camera **CAMERA_LIST, int nx, int ny, curandState *rand_state) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		curandState local_rand_state = *rand_state;
		*rand_state = local_rand_state;


		// SCENE 0
		// scene_0(HIT_LIST, HIT_WORLD, rand_state);
		// SCENE 1
		// scene_1(HIT_LIST, HIT_WORLD, rand_state);
		// SCENE 2
		scene_2(HIT_LIST, HIT_WORLD, rand_state);
		

		// CAMERA PLACEMENT FOR SCENE 0


		// vec3 lookfrom(278, 278, -800);
		// vec3 lookat(278, 278, 0);
		// float dist_to_focus = 10.0; (lookfrom - lookat).length();
		// float aperture = 0.0;
		// float vfov = 40;


		// CAMERA PLACEMENT FOR SCENE 1 & SCENE 2
		
		vec3 lookfrom(13, 2, 3);
		vec3 lookat(0, 0, 0);
		float dist_to_focus = 10.0; (lookfrom - lookat).length();
		float aperture = 0.0;
		float vfov = 50;
		


		*CAMERA_LIST = new camera(lookfrom,
			lookat,
			vec3(0, 1, 0),
			vfov,
			float(nx) / float(ny),
			aperture,
			dist_to_focus,
			0.0,1.0);
	}
}

// FREEING UP THE MEMORY FOR THE CREATION OF THE ENVIRONMENT
__global__ void free_world(hitable **HIT_LIST, hitable **HIT_WORLD, camera **CAMERA_LIST) {
	for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
		delete ((sphere *)HIT_LIST[i])->mat_ptr;
		delete HIT_LIST[i];
	}
	delete *HIT_WORLD;
	delete *CAMERA_LIST;
}

int main() {
	// TX-> DEFINING THE THREAD SIZE
	int tx = 64; // 32 * 32 	// THREAD X
	int ty = 8;	 		// THREAD Y

	// height and width of an image
	int width = 512; //256
	int height = 512; //256
	int num_samples = 128; //8
	
	std::ofstream output_file;
	output_file.open("output.ppm");
	std::cerr << "Rendering a " << width << "x" << height << " image with " << num_samples << " samples per pixel ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = width * height;	// NUMBER OF PIXELS-> SIZE OF THE RESOLUTION
	size_t fb_size = num_pixels * sizeof(vec3); // DEFINING THE FRAME BUFFER


	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	curandState *d_rand_state;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
	curandState *d_rand_state2;
	checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

	// we need that 2nd random state to be initialized for the world creation
	rand_init << <1, 1 >> > (d_rand_state2);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	// DEFINE THE HITABLES AND CAMERA LIST 

	hitable **HIT_LIST;

	int num_hitables = 22 * 22 + 1 + 3;
	checkCudaErrors(cudaMalloc((void **)&HIT_LIST, num_hitables * sizeof(hitable *)));

	hitable **HIT_WORLD;
	checkCudaErrors(cudaMalloc((void **)&HIT_WORLD, sizeof(hitable *)));

	camera **CAMERA_LIST;
	checkCudaErrors(cudaMalloc((void **)&CAMERA_LIST, sizeof(camera *)));

	create_world << <1, 1 >> > (HIT_LIST, HIT_WORLD, CAMERA_LIST, width, height, d_rand_state2);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	clock_t start, stop;
	std::cerr << "STARTING THE RENDERER ................\n";
	start = clock();
	// DEFINING THE BLOCKS DIMENSIONS
	dim3 blocks(width / tx + 1, height / ty + 1);
	// DEFINING THE THREAD DIMENSIONS
	dim3 threads(tx, ty);

	// RENDERING PROCES....
	render_init << <blocks, threads >> > (width, height, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	render << <blocks, threads >> > (fb, width, height, num_samples, CAMERA_LIST, HIT_WORLD, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	stop = clock();

	double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
	std::cerr << "took " << timer_seconds << " seconds.\n";


	// RENDERING THE FINAL OUTPUT AS IMAGE
	output_file << "P3\n" << width << " " << height << "\n255\n";
	for (int j = height - 1; j >= 0; j--) {

		for (int i = 0; i < width; i++) {

			size_t pixel_index = j * width + i;
			// red pixels
			int INT_R = int(255.99*fb[pixel_index].r());
			// green pixels
			int INT_G = int(255.99*fb[pixel_index].g());
			// blue pixels
			int INT_B = int(255.99*fb[pixel_index].b());

			output_file << INT_R << " " << INT_G << " " << INT_B << "\n";
		}
	}

	output_file.close();

	// clean up

	checkCudaErrors(cudaDeviceSynchronize());

	free_world << <1, 1 >> > (HIT_LIST, HIT_WORLD, CAMERA_LIST);

	checkCudaErrors(cudaGetLastError());
	// checkCudaErrors(cudaFree(CAMERA_LIST));
	checkCudaErrors(cudaFree(HIT_WORLD));
	checkCudaErrors(cudaFree(HIT_LIST));
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(fb));

	cudaDeviceReset();
}