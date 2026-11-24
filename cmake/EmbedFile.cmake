#! Writes SOURCE, which defines the array OpenDlssRr::Embedded::<NAME> with the bytes of SPIRV.
file(READ "${SPIRV}" bytes HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "std::byte{0x\\1}," bytes "${bytes}")
file(WRITE "${SOURCE}" "#include <cstddef>\n#include <span>\nnamespace OpenDlssRr::Embedded {\nnamespace {\nconstexpr std::byte data[] = {${bytes}};\n}\nextern const std::span<const std::byte> ${NAME}{data};\n}\n")
