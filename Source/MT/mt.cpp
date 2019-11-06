#include "mt.h"
#include <string>

void add_mt_commands(std::map<std::string, std::function<int(int, char **)>> &commands) {
    commands["lineshape"]  = &lineshape_main;
    commands["lorentzian"] = &lorentzian_main;
}