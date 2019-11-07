#include "Log.h"

#include "B1/b1.h"
#include "MT/mt.h"

using MainFunc = std::function<int(int, char **)>;

int main(int argc, char **argv) {

    std::map<std::string, MainFunc> commands;

#ifdef BUILD_B1
    add_b1_commands(commands);
#endif
#ifdef BUILD_MT
    add_mt_commands(commands);
#endif

    auto print_command_list = [&commands]() {
        QI::Log(true, "Available commands:");
        for (auto const &kv_pair : commands) {
            QI::Log(true, "\t{}", kv_pair.first);
        }
    };

    if (argc < 2) {
        print_command_list();
        QI::Fail("Must specify command");
    }

    std::string command(argv[1]);
    auto        find_command = commands.find(command);
    if (find_command == commands.end()) {
        print_command_list();
        QI::Fail("Unknown command {}", command);
    }
    return find_command->second(argc - 1, argv + 1);
}