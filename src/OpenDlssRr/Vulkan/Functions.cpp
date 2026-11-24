#include "OpenDlssRr/Vulkan/Functions.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace OpenDlssRr::Vulkan {

namespace {

#ifdef _WIN32
constexpr const char* loaderName = "vulkan-1.dll";

PFN_vkGetInstanceProcAddr openLoader() {
    static const HMODULE library = LoadLibraryA(loaderName);
    return library ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(reinterpret_cast<void*>(GetProcAddress(library, "vkGetInstanceProcAddr"))) : nullptr;
}
#else
constexpr const char* loaderName = "libvulkan.so.1";

PFN_vkGetInstanceProcAddr openLoader() {
    static void* const library = dlopen(loaderName, RTLD_NOW | RTLD_LOCAL);
    return library ? reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(library, "vkGetInstanceProcAddr")) : nullptr;
}
#endif

}

std::expected<void, Error> loadFunctions(const VkInstance instance, PFN_vkGetInstanceProcAddr getInstanceProcAddr) {
    if (!getInstanceProcAddr)
        getInstanceProcAddr = vkGetInstanceProcAddr ? vkGetInstanceProcAddr : openLoader();
    if (!getInstanceProcAddr)
        return fail(std::string("cannot open the Vulkan loader ") + loaderName);
    vkGetInstanceProcAddr = getInstanceProcAddr;
    if (!instance)
        return fail("no Vulkan instance");
#define OPENDLSS_RR_LOAD_VULKAN_FUNCTION(name)                                                                                                                 \
    name = reinterpret_cast<PFN_##name>(getInstanceProcAddr(instance, #name));                                                                                 \
    if (!name)                                                                                                                                                 \
        return fail("the Vulkan instance has no " #name);
    OPENDLSS_RR_VULKAN_INSTANCE_FUNCTIONS(OPENDLSS_RR_LOAD_VULKAN_FUNCTION)
#undef OPENDLSS_RR_LOAD_VULKAN_FUNCTION
    vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
        getInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    return {};
}

}
