/* In physics, the n-body simulation is the simulation of the motion of a set of M 
different objects due to the gravitational interaction among them. The following
program implement an N-body simulation. Each object to be simulated is modeled 
by means of the Body_t struct that contains fields for the position and the speed 
in a 3D space; data of N different objects is stored in an Body_t array.
The simulation starts by computing randomly  the initial position and speed of each
object. Then a number of time steps are simulated; at each time step the bodyForce()
function computes the new speed of each object due to the gravitational interaction 
and updateBodyPositions() functions computes the new position of each object.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define SOFTENING 1e-9f

typedef struct { 
  float x, y, z, vx, vy, vz; 
} Body_t;

double get_time();
void randomizeBodies(float *data, int n);
void bodyForce(Body_t *p, float dt, int n);
void updateBodyPositions(Body_t *p, float dt, int n);


void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body_t *p, float dt, int n) {
  for (int i = 0; i < n; i++) { 
    float Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; 
      Fy += dy * invDist3; 
      Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; 
    p[i].vy += dt*Fy; 
    p[i].vz += dt*Fz;
  }
}

void updateBodyPositions(Body_t *p, float dt, int n) {
  for (int i = 0 ; i < n; i++) { 
    p[i].x += p[i].vx*dt;
    p[i].y += p[i].vy*dt;
    p[i].z += p[i].vz*dt;
  }  
}

int main(const int argc, const char** argv) {
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 10;  // simulation iterations

  double cpu_start, cpu_end;

  Body_t *p = (Body_t*)malloc(nBodies*sizeof(Body_t));

  randomizeBodies((float*)p, 6*nBodies); // Init position / speed data

  cpu_start = get_time();
  for (int iter = 1; iter <= nIters; iter++) {
    bodyForce(p, dt, nBodies); // compute interbody forces
    updateBodyPositions(p, dt, nBodies); // integrate position
  }
  cpu_end = get_time();
  double avgTime = (cpu_end - cpu_start) / nIters; 

  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
  free(p);

  return 0;
}

// function to get the time of day in seconds
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

