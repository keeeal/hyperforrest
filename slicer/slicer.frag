#version 430

uniform mat4 p3d_ModelViewMatrix;

// Access the material attributes assigned via a Material object.
uniform struct {
  vec4 ambient;
  vec4 diffuse;
  vec4 emission;
  vec3 specular;
  float shininess;

  vec4 baseColor;
  float roughness;
  float metallic;
  float refractiveIndex;
} p3d_Material;

// The sum of all active ambient light colors.
uniform struct {
  vec4 ambient;
} p3d_LightModel;

// Contains information for each non-ambient light.
uniform struct p3d_LightSourceParameters {
  // Primary light color.
  vec4 color;

  // View-space position. If w=0, this is a directional light, with the xyz
  // being -direction.
  vec4 position;

  // Spotlight-only settings
  vec3 spotDirection;
  float spotExponent;
  float spotCutoff;
  float spotCosCutoff;

  // Individual attenuation constants
  float constantAttenuation;
  float linearAttenuation;
  float quadraticAttenuation;

  // constant, linear, quadratic attenuation in one vector
  vec3 attenuation;

  // Shadow map for this light source
  sampler2DShadow shadowMap;

  // Transforms view-space coordinates to shadow map coordinates
  mat4 shadowViewMatrix;
} p3d_LightSource[];

// Contains fog state.
uniform struct p3d_FogParameters {
  vec4 color;
  float density;
  float start;
  float end;
  float scale; // 1.0 / (end - start)
} p3d_Fog;

in FragmentData
{
    vec4 vertex;
    vec3 normal;
    vec4 colour;
} fragment_data;

out vec4 result;

void main()
{
    // ambient
    vec4 ambient = p3d_LightModel.ambient;

    // diffuse
    vec3 norm = normalize(fragment_data.normal);
    vec3 lightDir = normalize(p3d_LightSource[0].position.xyz - fragment_data.vertex.xyz);
    float diff = max(dot(norm, lightDir), 0.0);
    vec4 diffuse = diff * p3d_LightSource[0].color;

    // specular
    float specular_strength = .5;
    vec3 viewDir = normalize(-fragment_data.vertex.xyz);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    vec4 specular = specular_strength * spec * p3d_LightSource[0].color;

    // combine
    result = (ambient + diffuse + specular) * fragment_data.colour;
}