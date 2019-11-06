#include "Log.h"

#ifdef BUILD_MT
#include "MT/mt.h"
#endif

int main(int argc, char **argv) {

    std::map<std::string, std::function<int(int, char **)>> commands;
#ifdef BUILD_MT
    add_mt_commands(commands);
#endif
    if (argc < 2) {
        QI::Log(true, "Available commands:");
        for (auto const &kv_pair : commands) {
            QI::Log(true, "{}", kv_pair.first);
        }
        QI::Fail("Must specify sub-command");
    }

    std::string command(argv[1]);
    return commands[command](argc - 1, argv + 1);
}