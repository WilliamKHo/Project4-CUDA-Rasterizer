/**
 * @file      rasterize.cu
 * @brief     CUDA-accelerated rasterization pipeline.
 * @authors   Skeleton code: Yining Karl Li, Kai Ninomiya, Shuai Shao (Shrek)
 * @date      2012-2016
 * @copyright University of Pennsylvania & STUDENT
 */

#include <cmath>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/random.h>
#include <util/checkCUDAError.h>
#include <util/tiny_gltf_loader.h>
#include "rasterizeTools.h"
#include "rasterize.h"
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#define TILERENDER 0
#define TILERENDERWITHPREPROCESS 0
//These need to be defined at compile time, but they need to be mathematically sound. TILEX * TILEY = TILESIZE
#define TILEX 16
#define TILEY 16
#define TILESIZE 256

namespace {

	typedef unsigned short VertexIndex;
	typedef glm::vec3 VertexAttributePosition;
	typedef glm::vec3 VertexAttributeNormal;
	typedef glm::vec2 VertexAttributeTexcoord;
	typedef unsigned char TextureData;

	typedef unsigned char BufferByte;

	enum PrimitiveType{
		Point = 1,
		Line = 2,
		Triangle = 3
	};

	struct VertexOut {
		glm::vec4 pos;

		// TODO: add new attributes to your VertexOut
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		 glm::vec3 eyePos;	// eye space position used for shading
		 glm::vec3 eyeNor;	// eye space normal used for shading, cuz normal will go wrong after perspective transformation
		// glm::vec3 col;
		 glm::vec2 texcoord0;
		 TextureData* dev_diffuseTex = NULL;
		// int texWidth, texHeight;
		// ...
	};

	struct Primitive {
		PrimitiveType primitiveType = Triangle;	// C++ 11 init
		VertexOut v[3];
	};

	struct Fragment {
		glm::vec3 color;

		// TODO: add new attributes to your Fragment
		// The attributes listed below might be useful, 
		// but always feel free to modify on your own

		// glm::vec3 eyePos;	// eye space position used for shading
		// glm::vec3 eyeNor;
		// VertexAttributeTexcoord texcoord0;
		// TextureData* dev_diffuseTex;
		// ...
	};

	struct PrimitiveDevBufPointers {
		int primitiveMode;	//from tinygltfloader macro
		PrimitiveType primitiveType;
		int numPrimitives;
		int numIndices;
		int numVertices;

		// Vertex In, const after loaded
		VertexIndex* dev_indices;
		VertexAttributePosition* dev_position;
		VertexAttributeNormal* dev_normal;
		VertexAttributeTexcoord* dev_texcoord0;

		// Materials, add more attributes when needed
		TextureData* dev_diffuseTex;
		int diffuseTexWidth;
		int diffuseTexHeight;
		// TextureData* dev_specularTex;
		// TextureData* dev_normalTex;
		// ...

		// Vertex Out, vertex used for rasterization, this is changing every frame
		VertexOut* dev_verticesOut;

		// TODO: add more attributes when needed
	};

#if TILERENDERWITHPREPROCESS
	struct Tile {
		int numTriangles = 0;
		int triangleIndices[1000];
	};
#endif

}

static std::map<std::string, std::vector<PrimitiveDevBufPointers>> mesh2PrimitivesMap;


static int width = 0;
static int height = 0;
static float depthRange = 5000000.0f;

static int totalNumPrimitives = 0;
static Primitive *dev_primitives = NULL;
static Fragment *dev_fragmentBuffer = NULL;
static glm::vec3 *dev_framebuffer = NULL;

#if TILERENDERWITHPREPROCESS
static int *dev_triangleIndicesForFrag = NULL;
static Tile *dev_tiles = NULL;
#endif

static int * dev_depth = NULL;	// you might need this buffer when doing depth test
static int * mutex = NULL;

/**
 * Kernel that writes the image to the OpenGL PBO directly.
 */
__global__ 
void sendImageToPBO(uchar4 *pbo, int w, int h, glm::vec3 *image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        glm::vec3 color;
        color.x = glm::clamp(image[index].x, 0.0f, 1.0f) * 255.0;
        color.y = glm::clamp(image[index].y, 0.0f, 1.0f) * 255.0;
        color.z = glm::clamp(image[index].z, 0.0f, 1.0f) * 255.0;
        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

/** 
* Writes fragment colors to the framebuffer
*/
__global__
void render(int w, int h, Fragment *fragmentBuffer, glm::vec3 *framebuffer) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * w);

    if (x < w && y < h) {
        framebuffer[index] = fragmentBuffer[index].color;

		// TODO: add your fragment shader code here

    }
}

/**
 * Called once at the beginning of the program to allocate memory.
 */
void rasterizeInit(int w, int h) {
    width = w;
    height = h;
	cudaFree(dev_fragmentBuffer);
	cudaMalloc(&dev_fragmentBuffer, width * height * sizeof(Fragment));
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
    cudaFree(dev_framebuffer);
    cudaMalloc(&dev_framebuffer,   width * height * sizeof(glm::vec3));
    cudaMemset(dev_framebuffer, 0, width * height * sizeof(glm::vec3));
    
	cudaFree(dev_depth);
	cudaMalloc(&dev_depth, width * height * sizeof(int));

	cudaFree(mutex);
	cudaMalloc(&mutex, sizeof(int));

#if TILERENDERWITHPREPROCESS
	cudaFree(dev_triangleIndicesForFrag);
	cudaMalloc(&dev_triangleIndicesForFrag, width * height * sizeof(int));

	cudaFree(dev_tiles);
	cudaMalloc(&dev_tiles, ((width + TILEX - 1) / TILEX) * ((height + TILEY - 1) / TILEY) * sizeof(Tile));
#endif

	checkCUDAError("rasterizeInit");
}

__global__
void initDepth(int w, int h, int * depth)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < w && y < h)
	{
		int index = x + (y * w);
		depth[index] = INT_MAX;
	}
}


/**
* kern function with support for stride to sometimes replace cudaMemcpy
* One thread is responsible for copying one component
*/
__global__ 
void _deviceBufferCopy(int N, BufferByte* dev_dst, const BufferByte* dev_src, int n, int byteStride, int byteOffset, int componentTypeByteSize) {
	
	// Attribute (vec3 position)
	// component (3 * float)
	// byte (4 * byte)

	// id of component
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (i < N) {
		int count = i / n;
		int offset = i - count * n;	// which component of the attribute

		for (int j = 0; j < componentTypeByteSize; j++) {
			
			dev_dst[count * componentTypeByteSize * n 
				+ offset * componentTypeByteSize 
				+ j]

				= 

			dev_src[byteOffset 
				+ count * (byteStride == 0 ? componentTypeByteSize * n : byteStride) 
				+ offset * componentTypeByteSize 
				+ j];
		}
	}
	

}

__global__
void _nodeMatrixTransform(
	int numVertices,
	VertexAttributePosition* position,
	VertexAttributeNormal* normal,
	glm::mat4 MV, glm::mat3 MV_normal) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		position[vid] = glm::vec3(MV * glm::vec4(position[vid], 1.0f));
		normal[vid] = glm::normalize(MV_normal * normal[vid]);
	}
}

glm::mat4 getMatrixFromNodeMatrixVector(const tinygltf::Node & n) {
	
	glm::mat4 curMatrix(1.0);

	const std::vector<double> &m = n.matrix;
	if (m.size() > 0) {
		// matrix, copy it

		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				curMatrix[i][j] = (float)m.at(4 * i + j);
			}
		}
	} else {
		// no matrix, use rotation, scale, translation

		if (n.translation.size() > 0) {
			curMatrix[3][0] = n.translation[0];
			curMatrix[3][1] = n.translation[1];
			curMatrix[3][2] = n.translation[2];
		}

		if (n.rotation.size() > 0) {
			glm::mat4 R;
			glm::quat q;
			q[0] = n.rotation[0];
			q[1] = n.rotation[1];
			q[2] = n.rotation[2];

			R = glm::mat4_cast(q);
			curMatrix = curMatrix * R;
		}

		if (n.scale.size() > 0) {
			curMatrix = curMatrix * glm::scale(glm::vec3(n.scale[0], n.scale[1], n.scale[2]));
		}
	}

	return curMatrix;
}

void traverseNode (
	std::map<std::string, glm::mat4> & n2m,
	const tinygltf::Scene & scene,
	const std::string & nodeString,
	const glm::mat4 & parentMatrix
	) 
{
	const tinygltf::Node & n = scene.nodes.at(nodeString);
	glm::mat4 M = parentMatrix * getMatrixFromNodeMatrixVector(n);
	n2m.insert(std::pair<std::string, glm::mat4>(nodeString, M));

	auto it = n.children.begin();
	auto itEnd = n.children.end();

	for (; it != itEnd; ++it) {
		traverseNode(n2m, scene, *it, M);
	}
}

void rasterizeSetBuffers(const tinygltf::Scene & scene) {

	totalNumPrimitives = 0;

	std::map<std::string, BufferByte*> bufferViewDevPointers;

	// 1. copy all `bufferViews` to device memory
	{
		std::map<std::string, tinygltf::BufferView>::const_iterator it(
			scene.bufferViews.begin());
		std::map<std::string, tinygltf::BufferView>::const_iterator itEnd(
			scene.bufferViews.end());

		for (; it != itEnd; it++) {
			const std::string key = it->first;
			const tinygltf::BufferView &bufferView = it->second;
			if (bufferView.target == 0) {
				continue; // Unsupported bufferView.
			}

			const tinygltf::Buffer &buffer = scene.buffers.at(bufferView.buffer);

			BufferByte* dev_bufferView;
			cudaMalloc(&dev_bufferView, bufferView.byteLength);
			cudaMemcpy(dev_bufferView, &buffer.data.front() + bufferView.byteOffset, bufferView.byteLength, cudaMemcpyHostToDevice);

			checkCUDAError("Set BufferView Device Mem");

			bufferViewDevPointers.insert(std::make_pair(key, dev_bufferView));

		}
	}



	// 2. for each mesh: 
	//		for each primitive: 
	//			build device buffer of indices, materail, and each attributes
	//			and store these pointers in a map
	{

		std::map<std::string, glm::mat4> nodeString2Matrix;
		auto rootNodeNamesList = scene.scenes.at(scene.defaultScene);

		{
			auto it = rootNodeNamesList.begin();
			auto itEnd = rootNodeNamesList.end();
			for (; it != itEnd; ++it) {
				traverseNode(nodeString2Matrix, scene, *it, glm::mat4(1.0f));
			}
		}


		// parse through node to access mesh

		auto itNode = nodeString2Matrix.begin();
		auto itEndNode = nodeString2Matrix.end();
		for (; itNode != itEndNode; ++itNode) {

			const tinygltf::Node & N = scene.nodes.at(itNode->first);
			const glm::mat4 & matrix = itNode->second;
			const glm::mat3 & matrixNormal = glm::transpose(glm::inverse(glm::mat3(matrix)));

			auto itMeshName = N.meshes.begin();
			auto itEndMeshName = N.meshes.end();

			for (; itMeshName != itEndMeshName; ++itMeshName) {

				const tinygltf::Mesh & mesh = scene.meshes.at(*itMeshName);

				auto res = mesh2PrimitivesMap.insert(std::pair<std::string, std::vector<PrimitiveDevBufPointers>>(mesh.name, std::vector<PrimitiveDevBufPointers>()));
				std::vector<PrimitiveDevBufPointers> & primitiveVector = (res.first)->second;

				// for each primitive
				for (size_t i = 0; i < mesh.primitives.size(); i++) {
					const tinygltf::Primitive &primitive = mesh.primitives[i];

					if (primitive.indices.empty())
						return;

					// TODO: add new attributes for your PrimitiveDevBufPointers when you add new attributes
					VertexIndex* dev_indices = NULL;
					VertexAttributePosition* dev_position = NULL;
					VertexAttributeNormal* dev_normal = NULL;
					VertexAttributeTexcoord* dev_texcoord0 = NULL;

					// ----------Indices-------------

					const tinygltf::Accessor &indexAccessor = scene.accessors.at(primitive.indices);
					const tinygltf::BufferView &bufferView = scene.bufferViews.at(indexAccessor.bufferView);
					BufferByte* dev_bufferView = bufferViewDevPointers.at(indexAccessor.bufferView);

					// assume type is SCALAR for indices
					int n = 1;
					int numIndices = indexAccessor.count;
					int componentTypeByteSize = sizeof(VertexIndex);
					int byteLength = numIndices * n * componentTypeByteSize;

					dim3 numThreadsPerBlock(128);
					dim3 numBlocks((numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					cudaMalloc(&dev_indices, byteLength);
					_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
						numIndices,
						(BufferByte*)dev_indices,
						dev_bufferView,
						n,
						indexAccessor.byteStride,
						indexAccessor.byteOffset,
						componentTypeByteSize);


					checkCUDAError("Set Index Buffer");


					// ---------Primitive Info-------

					// Warning: LINE_STRIP is not supported in tinygltfloader
					int numPrimitives;
					PrimitiveType primitiveType;
					switch (primitive.mode) {
					case TINYGLTF_MODE_TRIANGLES:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices / 3;
						break;
					case TINYGLTF_MODE_TRIANGLE_STRIP:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_TRIANGLE_FAN:
						primitiveType = PrimitiveType::Triangle;
						numPrimitives = numIndices - 2;
						break;
					case TINYGLTF_MODE_LINE:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices / 2;
						break;
					case TINYGLTF_MODE_LINE_LOOP:
						primitiveType = PrimitiveType::Line;
						numPrimitives = numIndices + 1;
						break;
					case TINYGLTF_MODE_POINTS:
						primitiveType = PrimitiveType::Point;
						numPrimitives = numIndices;
						break;
					default:
						// output error
						break;
					};


					// ----------Attributes-------------

					auto it(primitive.attributes.begin());
					auto itEnd(primitive.attributes.end());

					int numVertices = 0;
					// for each attribute
					for (; it != itEnd; it++) {
						const tinygltf::Accessor &accessor = scene.accessors.at(it->second);
						const tinygltf::BufferView &bufferView = scene.bufferViews.at(accessor.bufferView);

						int n = 1;
						if (accessor.type == TINYGLTF_TYPE_SCALAR) {
							n = 1;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC2) {
							n = 2;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC3) {
							n = 3;
						}
						else if (accessor.type == TINYGLTF_TYPE_VEC4) {
							n = 4;
						}

						BufferByte * dev_bufferView = bufferViewDevPointers.at(accessor.bufferView);
						BufferByte ** dev_attribute = NULL;

						numVertices = accessor.count;
						int componentTypeByteSize;

						// Note: since the type of our attribute array (dev_position) is static (float32)
						// We assume the glTF model attribute type are 5126(FLOAT) here

						if (it->first.compare("POSITION") == 0) {
							componentTypeByteSize = sizeof(VertexAttributePosition) / n;
							dev_attribute = (BufferByte**)&dev_position;
						}
						else if (it->first.compare("NORMAL") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeNormal) / n;
							dev_attribute = (BufferByte**)&dev_normal;
						}
						else if (it->first.compare("TEXCOORD_0") == 0) {
							componentTypeByteSize = sizeof(VertexAttributeTexcoord) / n;
							dev_attribute = (BufferByte**)&dev_texcoord0;
						}

						std::cout << accessor.bufferView << "  -  " << it->second << "  -  " << it->first << '\n';

						dim3 numThreadsPerBlock(128);
						dim3 numBlocks((n * numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
						int byteLength = numVertices * n * componentTypeByteSize;
						cudaMalloc(dev_attribute, byteLength);

						_deviceBufferCopy << <numBlocks, numThreadsPerBlock >> > (
							n * numVertices,
							*dev_attribute,
							dev_bufferView,
							n,
							accessor.byteStride,
							accessor.byteOffset,
							componentTypeByteSize);

						std::string msg = "Set Attribute Buffer: " + it->first;
						checkCUDAError(msg.c_str());
					}

					// malloc for VertexOut
					VertexOut* dev_vertexOut;
					cudaMalloc(&dev_vertexOut, numVertices * sizeof(VertexOut));
					checkCUDAError("Malloc VertexOut Buffer");

					// ----------Materials-------------

					// You can only worry about this part once you started to 
					// implement textures for your rasterizer
					TextureData* dev_diffuseTex = NULL;
					int diffuseTexWidth = 0;
					int diffuseTexHeight = 0;
					if (!primitive.material.empty()) {
						const tinygltf::Material &mat = scene.materials.at(primitive.material);
						printf("material.name = %s\n", mat.name.c_str());

						if (mat.values.find("diffuse") != mat.values.end()) {
							std::string diffuseTexName = mat.values.at("diffuse").string_value;
							if (scene.textures.find(diffuseTexName) != scene.textures.end()) {
								const tinygltf::Texture &tex = scene.textures.at(diffuseTexName);
								if (scene.images.find(tex.source) != scene.images.end()) {
									const tinygltf::Image &image = scene.images.at(tex.source);

									size_t s = image.image.size() * sizeof(TextureData);
									cudaMalloc(&dev_diffuseTex, s);
									cudaMemcpy(dev_diffuseTex, &image.image.at(0), s, cudaMemcpyHostToDevice);
									
									diffuseTexWidth = image.width;
									diffuseTexHeight = image.height;

									checkCUDAError("Set Texture Image data");
								}
							}
						}

						// TODO: write your code for other materails
						// You may have to take a look at tinygltfloader
						// You can also use the above code loading diffuse material as a start point 
					}


					// ---------Node hierarchy transform--------
					cudaDeviceSynchronize();
					
					dim3 numBlocksNodeTransform((numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
					_nodeMatrixTransform << <numBlocksNodeTransform, numThreadsPerBlock >> > (
						numVertices,
						dev_position,
						dev_normal,
						matrix,
						matrixNormal);

					checkCUDAError("Node hierarchy transformation");

					// at the end of the for loop of primitive
					// push dev pointers to map
					primitiveVector.push_back(PrimitiveDevBufPointers{
						primitive.mode,
						primitiveType,
						numPrimitives,
						numIndices,
						numVertices,

						dev_indices,
						dev_position,
						dev_normal,
						dev_texcoord0,

						dev_diffuseTex,
						diffuseTexWidth,
						diffuseTexHeight,

						dev_vertexOut	//VertexOut
					});

					totalNumPrimitives += numPrimitives;

				} // for each primitive

			} // for each mesh

		} // for each node

	}
	

	// 3. Malloc for dev_primitives
	{
		cudaMalloc(&dev_primitives, totalNumPrimitives * sizeof(Primitive));
	}
	

	// Finally, cudaFree raw dev_bufferViews
	{

		std::map<std::string, BufferByte*>::const_iterator it(bufferViewDevPointers.begin());
		std::map<std::string, BufferByte*>::const_iterator itEnd(bufferViewDevPointers.end());
			
			//bufferViewDevPointers

		for (; it != itEnd; it++) {
			cudaFree(it->second);
		}

		checkCUDAError("Free BufferView Device Mem");
	}


}



__global__ 
void _vertexTransformAndAssembly(
	int numVertices, 
	PrimitiveDevBufPointers primitive, 
	glm::mat4 MVP, glm::mat4 MV, glm::mat3 MV_normal, 
	int width, int height) {

	// vertex id
	int vid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (vid < numVertices) {
		//VertexIndex vertexIndex = primitive.dev_indices[vid];
		VertexOut &ref_vs_output = primitive.dev_verticesOut[vid];
		//
		//primitive.dev_verticesOut[vid].texcoord0 = primitive.dev_texcoord0[vid];
		glm::vec4 pos = glm::vec4(primitive.dev_position[vid], 1.0f);
		ref_vs_output.pos = MVP * pos;
		ref_vs_output.eyePos = glm::vec3(MV * pos);
		ref_vs_output.eyeNor = glm::vec3(MV_normal * primitive.dev_normal[vid]);

		//ref_vs_output.eyeNor = primitive.dev_normal[vid];
		//ref_vs_output.dev_diffuseTex = primitive.dev_diffuseTex;

		// TODO: Apply vertex transformation here
		// Multiply the MVP matrix for each vertex position, this will transform everything into clipping space
		// Then divide the pos by its w element to transform into NDC space
		// Finally transform x and y to viewport space

		//basic 

		// TODO: Apply vertex assembly here
		// Assemble all attribute arraies into the primitive array
		
	}
}



static int curPrimitiveBeginId = 0;

__global__ 
void _primitiveAssembly(int numIndices, int curPrimitiveBeginId, Primitive* dev_primitives, PrimitiveDevBufPointers primitive) {

	// index id
	int iid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (iid < numIndices) {

		// TODO: uncomment the following code for a start
		// This is primitive assembly for triangles

		int pid;	// id for cur primitives vector
		if (primitive.primitiveMode == TINYGLTF_MODE_TRIANGLES) {
			pid = iid / (int)primitive.primitiveType;
			dev_primitives[pid + curPrimitiveBeginId].v[iid % (int)primitive.primitiveType]
				= primitive.dev_verticesOut[primitive.dev_indices[iid]];
		}


		// TODO: other primitive types (point, line)
	}
	
}

__device__
glm::vec3 getBarycentricWeights(glm::vec3 p, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3) {
	float totalArea = glm::length(glm::cross(p1 - p3, p2 - p3)) / 2.0f;

	float area1 = glm::length(glm::cross(p2 - p, p3 - p)) / 2.0f;
	float area2 = glm::length(glm::cross(p3 - p, p1 - p)) / 2.0f;
	float area3 = glm::length(glm::cross(p1 - p, p2 - p)) / 2.0f;

	return glm::vec3(area1 / totalArea, area2 / totalArea, area3 / totalArea);
} 

__device__ 
bool isInTriangle(glm::vec3 p, glm::vec3 p1 , glm::vec3 p2 , glm::vec3 p3 ) {
	float totalArea = glm::abs((p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2.0f);
	
	float area1 = glm::abs((p.x * (p2.y - p3.y) + p2.x * (p3.y - p.y) + p3.x * (p.y - p2.y)) / 2.0f);
	float area2 = glm::abs((p1.x * (p.y - p3.y) + p.x * (p3.y - p1.y) + p3.x * (p1.y - p.y)) / 2.0f);
	float area3 = glm::abs((p1.x * (p2.y - p.y) + p2.x * (p.y - p1.y) + p.x * (p1.y - p2.y)) / 2.0f);	
	glm::vec3 bw = glm::vec3(area1 / totalArea, area2 / totalArea, area3 / totalArea);
	return !((bw.x + bw.y + bw.z) > 1.00001f);
}

/**
* Computes the axis-aligned bounding box for a given prim
* Outputs:
* glm::ivec4 Contains pixel coordinates for left bottom and top right corners of AABB
*/
__device__
glm::ivec4 computeAABB(int width, int height, Primitive prim) {
	float maxX = fmaxf(prim.v[0].pos.x, fmaxf(prim.v[1].pos.x, prim.v[2].pos.x));
	float maxY = fmaxf(prim.v[0].pos.y, fmaxf(prim.v[1].pos.y, prim.v[2].pos.y));
	float minX = fminf(prim.v[0].pos.x, fminf(prim.v[1].pos.x, prim.v[2].pos.x));
	float minY = fminf(prim.v[0].pos.y, fminf(prim.v[1].pos.y, prim.v[2].pos.y));

	return glm::ivec4(
		(int) ((minX + 1.0f) * (width / 2)) - 1,
		(int) ((1.0f - maxY) * (height / 2)) - 1, //Necessary to flip max and min Y because in pixel space, 0,0 is the top left
		(int) ((maxX + 1.0f) * (width / 2)) + 1,
		(int) ((1.0f - minY) * (height / 2) + 1)
	);
}

/**
* Converts pixel coordinates to fragment index
*/
__device__
int pixelToFragIndex(int x, int y, int width, int height) {
	return y * width - x;
}

__device__
glm::vec3 NDCtoPixel(glm::vec3 p, int width, int height) {
	return glm::vec3((p.x + 1.0f) * (width / 2), (1.0f - p.y) * (height / 2), p.z);
}

/* Takes in information in NDC and outputs z-depth
*/

__device__
float computeFragmentDepth(glm::vec3 p, Primitive prim) {
	glm::vec3 p1 = glm::vec3(prim.v[0].pos);
	glm::vec3 p2 = glm::vec3(prim.v[1].pos);
	glm::vec3 p3 = glm::vec3(prim.v[2].pos);

	p1.z = 0.0f;
	p2.z = 0.0f;
	p3.z = 0.0f;

	glm::vec3 eyePos1 = glm::vec3(prim.v[0].eyePos);
	glm::vec3 eyePos2 = glm::vec3(prim.v[1].eyePos);
	glm::vec3 eyePos3 = glm::vec3(prim.v[2].eyePos);

	float totalArea = glm::length(glm::cross(p1 - p3, p2 - p3)) / 2.0f;

	float area1 = glm::length(glm::cross(p2 - p, p3 - p)) / 2.0f;
	float area2 = glm::length(glm::cross(p1 - p, p3 - p)) / 2.0f;
	float area3 = glm::length(glm::cross(p1 - p, p2 - p)) / 2.0f;
	glm::vec3 bw = glm::vec3(area1 / totalArea, area2 / totalArea, area3 / totalArea);

	return 1.0f / ((1.0f / eyePos1.z) * bw.x + (1.0f / eyePos2.z) * bw.y + (1.0f / eyePos3.z)  * bw.z);
}

__device__
glm::vec3 computeLambertian(glm::vec3 normal, glm::vec3 light, glm::vec3 baseColor) {
	return fmaxf(glm::dot(glm::normalize(normal), glm::normalize(light)), 0.1f) * baseColor;
}

__global__
void rasterizeTriangles(int numPrimitives, 
	int width,
	int height,
	Primitive* dev_primitives, 
	Fragment* dev_fragmentBuffer,
	int * dev_depth,
	float depthRange,
	int * mutex) {
	int primId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (primId < numPrimitives) {
		Primitive& primitive = dev_primitives[primId];
		glm::ivec4 AABB = computeAABB(width, height, primitive);
		glm::vec3 pPix1 = NDCtoPixel(glm::vec3(primitive.v[0].pos), width, height);
		glm::vec3 pPix2 = NDCtoPixel(glm::vec3(primitive.v[1].pos), width, height);
		glm::vec3 pPix3 = NDCtoPixel(glm::vec3(primitive.v[2].pos), width, height);
		glm::vec3 tri[3] = {
			pPix1, 
			pPix2, 
			pPix3 };
		for (int y = AABB.y; y < AABB.w; y++) {
			for (int x = AABB.x; x < AABB.z; x++) {
				glm::vec3 bw = calculateBarycentricCoordinate(tri, glm::vec2(x, y));
				if (isBarycentricCoordInBounds(bw)) {
					int fragIndex = pixelToFragIndex(x, y, width, height);
					//printf("%f\n", computeFragmentDepth(glm::vec3(x, y, 0.0f), primitive));
					int depth = (int) (depthRange * -getZAtCoordinate(bw, tri));
					//atomicMin(dev_depth + fragIndex, depth);
					//if (depth == dev_depth[fragIndex]) dev_fragmentBuffer[fragIndex].color = glm::vec3((float) primId / (float) numPrimitives, 0.0f, 0.0f);
					//printf("%i\n", depth);
					bool isSet;
					do {
						isSet = (atomicCAS(mutex, 0, 1) == 0);
						if (isSet) {
							dev_depth[fragIndex] = min(dev_depth[fragIndex], depth);
							if (depth == dev_depth[fragIndex]) dev_fragmentBuffer[fragIndex].color = primitive.v[0].eyeNor;

						}
						if (isSet) {
							*mutex = 0;
						}
					} while (!isSet);
				}
			}
		}
	}

}

__device__
int tileIndexToFragIndex(int tileIndex, int tileX, glm::ivec4 tile, int width, int height) {
	int tileYcoord = tileIndex / tileX;
	int tileXcoord = tileIndex - (tileYcoord * tileX);
	//printf("%i, %i, %i\n", tileIndex, tileX, tileYcoord);

	return pixelToFragIndex(tile.x + tileXcoord, tile.y + tileYcoord, width, height);
}

__device__
bool pixelInTile(int x, int y, glm::ivec4 tile) {
	return (x >= tile.x && x < tile.z && y >= tile.y && y < tile.w);
}

__global__
void tileRasterizeTriangles(int numPrimitives,
	int width,
	int height,
	Primitive* dev_primitives,
	Fragment* dev_fragmentBuffer,
	int * dev_depth,
	float depthRange,
	int * mutex) {
	//Allocate shared memory for tile
	//Format: x,y,z is color and w is depth
	// On Moore 100 Machines, max shared memory is c000 (49152) bytes, 
	// with this, shared memory usage is 40000 bytes
	int tileX = blockIdx.x;
	int tileY = blockIdx.y;
	__shared__ glm::vec4 shared_tileBuffer[TILESIZE];
	int sharedWritesPerThread = (TILESIZE + blockDim.x - 1) / blockDim.x;
	for (int i = 0; i < sharedWritesPerThread; i++) {
		if (i + threadIdx.x * sharedWritesPerThread < TILESIZE) {
			shared_tileBuffer[i + threadIdx.x * sharedWritesPerThread] = glm::vec4(0.0f, 0.0f, 0.0f, FLT_MAX);
		}
	}
	__syncthreads();
	glm::ivec4 tile = glm::ivec4(tileX * TILEX, tileY * TILEY, (tileX + 1) * TILEX, (tileY + 1) * TILEY);
	int primId = threadIdx.x;
	if (primId < numPrimitives) {
		//Initialize shared memory
		//Evaluate bounding box and determine if the triangle needs to be rendered in this tile
		Primitive& primitive = dev_primitives[primId];
		glm::ivec4 AABB = computeAABB(width, height, primitive);
		if (AABB.x >= tile.z || AABB.z < tile.x || AABB.y >= tile.w || AABB.w < tile.y) {
			return;
		}

		glm::vec3 pPix1 = NDCtoPixel(glm::vec3(primitive.v[0].pos), width, height);
		glm::vec3 pPix2 = NDCtoPixel(glm::vec3(primitive.v[1].pos), width, height);
		glm::vec3 pPix3 = NDCtoPixel(glm::vec3(primitive.v[2].pos), width, height);
		glm::vec3 tri[3] = {
			pPix1,
			pPix2,
			pPix3 };
		for (int y = AABB.y; y < AABB.w; y++) {
			for (int x = AABB.x; x < AABB.z; x++) {
				//Test if pixel x,y is in the triangle
				glm::vec3 bw = calculateBarycentricCoordinate(tri, glm::vec2(x, y));
				if (isBarycentricCoordInBounds(bw) && pixelInTile(x, y, tile)) {
					//Convert the pixel to a fragmentIndex and compute its depth
					int fragIndex = pixelToFragIndex(x, y, width, height);
					int fragTileIndex = (x - tile.x) + (y - tile.y) * TILEY;
					//Cull fragments outside of tile
					float depth = (depthRange * -getZAtCoordinate(bw, tri));
					//depth test, the mutex ensures that the code within "isSet" happens atomically
					//that is, that code is guaranteed to execute in a single thread without anything executing in
					//any other thread
					bool isSet;
					do {
						isSet = (atomicCAS(mutex, 0, 1) == 0);
						if (isSet) {
							/*dev_depth[fragIndex] = min(dev_depth[fragIndex], (int)depth);
							if (depth == dev_depth[fragIndex]) dev_fragmentBuffer[fragIndex].color = primitive.v[0].eyeNor;*/
							float originalDepth = shared_tileBuffer[fragTileIndex].w;
							shared_tileBuffer[fragTileIndex].w = fminf(originalDepth, depth);
							if (depth < originalDepth) {
								glm::vec3 color = primitive.v[0].eyeNor;
								shared_tileBuffer[fragTileIndex].x = color.x;
								shared_tileBuffer[fragTileIndex].y = color.y;
								shared_tileBuffer[fragTileIndex].z = color.z;

							}
						}
						if (isSet) {
							*mutex = 0;
						}
					} while (!isSet);
				}
			}
		}
		//This is therefore where I should call __syncThreads(); and write to the dev_fragmentBuffer
		//Will need some kind of for loop to write to the dev_fragmentBuffer
	}
	__syncthreads();

	//Write shared memory to fragmentbuffer. This might be more optimal if we wanted to retire threads after the above branch and split up writes
	//based on the number of primitives.
	for (int i = 0; i < sharedWritesPerThread; i++) {
		if (i + threadIdx.x * sharedWritesPerThread < TILESIZE) {
			int fragIndex = tileIndexToFragIndex(i + threadIdx.x * sharedWritesPerThread,
				TILEX,
				tile,
				width,
				height);
			dev_fragmentBuffer[fragIndex].color = glm::vec3(shared_tileBuffer[i + threadIdx.x * sharedWritesPerThread]);
		}
	}
}


#if TILERENDERWITHPREPROCESS
__global__
void computeTrianglesToBeRendered(int numPrimitives,
	int width,
	int height,
	int tileGridWidth,
	Primitive* dev_primitives,
	Tile* dev_tiles,
	int *mutex) {
	//Parellelize over triangles
	//Using AABB, bucket the triangles into the tiles
	int primId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (primId < numPrimitives) {
		Primitive& primitive = dev_primitives[primId];
		glm::ivec4 AABB = computeAABB(width, height, primitive);
		int minX = (AABB.x / TILEX) * TILEX;
		int minY = (AABB.y / TILEY) * TILEY;
		for (int y = minY; y < AABB.w; y += TILEY) {
			for (int x = minX; x < AABB.z; x += TILEX) {
				Tile& tile = dev_tiles[(x / TILEX) + (y / TILEY) * tileGridWidth];
				bool isSet;
				do {
					isSet = (atomicCAS(mutex, 0, 1) == 0);
					if (isSet) {
						int bucketIdx = tile.numTriangles;
						tile.triangleIndices[bucketIdx] = primId;
						tile.numTriangles++;
					}
					if (isSet) {
						*mutex = 0;
					}
				} while (!isSet);
			}
		}
	}

}

__global__
void tileRasterizeTrianglesAfterPreProcess(
	int width,
	int height,
	int tileGridWidth,
	Primitive* dev_primitives,
	Tile* dev_tiles,
	Fragment* dev_fragmentBuffer,
	float depthRange) {
	//Block is tile
	//Thread is pixel
	//Loop over the triangles in the bucket
	
	__shared__ glm::vec4 shared_tileBuffer[TILESIZE];

	int tileX = blockIdx.x;
	int tileY = blockIdx.y;
	int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

	int fragIdx = pixelToFragIndex(pixelX, pixelY, width, height);
	Tile& tile = dev_tiles[tileX + tileY * tileGridWidth];
	int idxInTile = threadIdx.x + threadIdx.y * blockDim.x;

	shared_tileBuffer[idxInTile] = glm::vec4(0.0f, 0.0f, 0.0f, FLT_MAX);

	__syncthreads();

	for (int i = 0; i < tile.numTriangles; i++) {
		int primId = tile.triangleIndices[i];
		Primitive& primitive = dev_primitives[primId];
		glm::vec3 pPix1 = NDCtoPixel(glm::vec3(primitive.v[0].pos), width, height);
		glm::vec3 pPix2 = NDCtoPixel(glm::vec3(primitive.v[1].pos), width, height);
		glm::vec3 pPix3 = NDCtoPixel(glm::vec3(primitive.v[2].pos), width, height);
		glm::vec3 tri[3] = {
			pPix1,
			pPix2,
			pPix3 };
		glm::ivec4 AABB = computeAABB(width, height, primitive);
		glm::vec3 bw = calculateBarycentricCoordinate(tri, glm::vec2(pixelX, pixelY));
		if (isBarycentricCoordInBounds(bw) && pixelInTile(pixelX, pixelY, glm::ivec4(tileX * TILEX, 
			tileY * TILEY, 
			(tileX + 1) * TILEX,
			(tileY + 1) * TILEY))) {
			float depth = (depthRange * -getZAtCoordinate(bw, tri));
			float originalDepth = shared_tileBuffer[idxInTile].w;
			shared_tileBuffer[idxInTile].w = fminf(originalDepth, depth);
			if (depth < originalDepth) {
				glm::vec3 color = primitive.v[0].eyeNor;
				shared_tileBuffer[idxInTile].x = color.x;
				shared_tileBuffer[idxInTile].y = color.y;
				shared_tileBuffer[idxInTile].z = color.z;
			}
		}
	}

	dev_fragmentBuffer[fragIdx].color = glm::vec3(shared_tileBuffer[idxInTile]);
}

__global__
void colorTileBorders(int width, int height, Fragment* dev_fragmentBuffer) {
	int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;

	if ((threadIdx.x == 0 || threadIdx.y == 0) && pixelX < width && pixelY < height) {
		dev_fragmentBuffer[pixelToFragIndex(pixelX, pixelY, width, height)].color = glm::vec3(1.0f, 0.0f, 0.0f);
	}
}
#endif

__global__
void shadeLambertian(int width, int height, Fragment* dev_fragmentBuffer, glm::vec3 light) {
	int pixelX = blockIdx.x * blockDim.x + threadIdx.x;
	int pixelY = blockIdx.y * blockDim.y + threadIdx.y;
	if (pixelX * pixelY < width * height) {
		int fragIdx = pixelToFragIndex(pixelX, pixelY, width, height);
		dev_fragmentBuffer[fragIdx].color = glm::dot(dev_fragmentBuffer[fragIdx].color, light) * glm::vec3(1.0f, 0.5f, 0.5f);
	}
}
__global__
void redFragments(int width, int height, Fragment* dev_fragmentBuffer) {
	int fragX = (blockIdx.x * blockDim.x) + threadIdx.x;
	int fragY = (blockIdx.y * blockDim.y) + threadIdx.y;

	int index = (width - fragX) + (height - fragY) * width; 
	if (index < width * height) {
		dev_fragmentBuffer[index].color = glm::vec3((float) fragX / (float) width, (float) fragY / (float) height, 0.0f);
	}
}

/**
 * Perform rasterization.
 */
void rasterize(uchar4 *pbo, const glm::mat4 & MVP, const glm::mat4 & MV, const glm::mat3 MV_normal) {
    int sideLength2d = 8;
    dim3 blockSize2d(sideLength2d, sideLength2d);
    dim3 blockCount2d((width  - 1) / blockSize2d.x + 1,
		(height - 1) / blockSize2d.y + 1);

	// Execute your rasterization pipeline here
	// (See README for rasterization pipeline outline.)

	// Vertex Process & primitive assembly
	{
		curPrimitiveBeginId = 0;
		dim3 numThreadsPerBlock(128);

		auto it = mesh2PrimitivesMap.begin();
		auto itEnd = mesh2PrimitivesMap.end();

		for (; it != itEnd; ++it) {
			auto p = (it->second).begin();	// each primitive
			auto pEnd = (it->second).end();
			for (; p != pEnd; ++p) {
				dim3 numBlocksForVertices((p->numVertices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);
				dim3 numBlocksForIndices((p->numIndices + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x);

				_vertexTransformAndAssembly << < numBlocksForVertices, numThreadsPerBlock >> >(p->numVertices, *p, MVP, MV, MV_normal, width, height);
				checkCUDAError("Vertex Processing");
				cudaDeviceSynchronize();
				_primitiveAssembly << < numBlocksForIndices, numThreadsPerBlock >> >
					(p->numIndices, 
					curPrimitiveBeginId, 
					dev_primitives, 
					*p);
				checkCUDAError("Primitive Assembly");

				curPrimitiveBeginId += p->numPrimitives;
			}
		}

		checkCUDAError("Vertex Processing and Primitive Assembly");
	}
	
	cudaMemset(dev_fragmentBuffer, 0, width * height * sizeof(Fragment));
	initDepth << <blockCount2d, blockSize2d >> >(width, height, dev_depth);

	dim3 threadsPerBlock((curPrimitiveBeginId + 31) / 32);
	dim3 numBlocksForRasterization((curPrimitiveBeginId + threadsPerBlock.x - 1) / threadsPerBlock.x);

	cudaMemset(mutex, 0, sizeof(int));

	glm::vec3 light = glm::vec3(1.0f, 1.0f, 1.0f);
	
	 //TODO: rasterize
#if TILERENDER
	dim3 blockDim2dTiles((width + TILEX - 1) / TILEX, (height + TILEY - 1) / TILEY);
	tileRasterizeTriangles << <blockDim2dTiles, 1024 >> >
		(curPrimitiveBeginId,
			width,
			height,
			dev_primitives,
			dev_fragmentBuffer,
			dev_depth,
			depthRange,
			mutex);
	/*tileRasterizeTriangles << <1, 128 >> >
		(curPrimitiveBeginId,
			width,
			height,
			dev_primitives,
			dev_fragmentBuffer,
			dev_depth,
			depthRange,
			mutex,
			1,
			1);*/

#elif TILERENDERWITHPREPROCESS

	computeTrianglesToBeRendered << <numBlocksForRasterization, threadsPerBlock >> > (
		curPrimitiveBeginId,
		width,
		height,
		(width + TILEX - 1) / TILEX,
		dev_primitives,
		dev_tiles,
		mutex);

	dim3 blockCountForTiles((width + TILEX - 1) / TILEX, (height + TILEY - 1) / TILEY);
	dim3 blockSizeForTiles(TILEX, TILEY);

	checkCUDAError("triangle bucketing");

	tileRasterizeTrianglesAfterPreProcess<<<blockCountForTiles, blockSizeForTiles>>>(
		width,
		height,
		(width + TILEX - 1) / TILEX,
		dev_primitives,
		dev_tiles,
		dev_fragmentBuffer,
		depthRange
	);

	shadeLambertian << <blockCount2d, blockSize2d >> > (
		width, height,
		dev_fragmentBuffer,
		light);

	colorTileBorders << <blockCount2d, blockSize2d >> > (
		width, 
		height,
		dev_fragmentBuffer);
#else
	rasterizeTriangles << <numBlocksForRasterization, threadsPerBlock >> > 
		(curPrimitiveBeginId, 
		width,
		height, 
		dev_primitives, 
		dev_fragmentBuffer,
		dev_depth,
		depthRange,
		mutex);

	/*shadeLambertian << <blockCount2d, blockSize2d >> > (
		width, height,
		dev_fragmentBuffer,
		light);*/

#endif
	
	//redFragments << <blockCount2d, blockSize2d >> > (width, height, dev_fragmentBuffer);
	checkCUDAError("rasterization");


    // Copy depthbuffer colors into framebuffer
	render << <blockCount2d, blockSize2d >> >(width, height, dev_fragmentBuffer, dev_framebuffer);
	checkCUDAError("fragment shader");
    // Copy framebuffer into OpenGL buffer for OpenGL previewing
    sendImageToPBO<<<blockCount2d, blockSize2d>>>(pbo, width, height, dev_framebuffer);
    checkCUDAError("copy render result to pbo");
}

/**
 * Called once at the end of the program to free CUDA memory.
 */
void rasterizeFree() {

    // deconstruct primitives attribute/indices device buffer

	auto it(mesh2PrimitivesMap.begin());
	auto itEnd(mesh2PrimitivesMap.end());
	for (; it != itEnd; ++it) {
		for (auto p = it->second.begin(); p != it->second.end(); ++p) {
			cudaFree(p->dev_indices);
			cudaFree(p->dev_position);
			cudaFree(p->dev_normal);
			cudaFree(p->dev_texcoord0);
			cudaFree(p->dev_diffuseTex);

			cudaFree(p->dev_verticesOut);

			
			//TODO: release other attributes and materials
		}
	}

	////////////

    cudaFree(dev_primitives);
    dev_primitives = NULL;

	cudaFree(dev_fragmentBuffer);
	dev_fragmentBuffer = NULL;

    cudaFree(dev_framebuffer);
    dev_framebuffer = NULL;

	cudaFree(dev_depth);
	dev_depth = NULL;

    checkCUDAError("rasterize Free");
}
