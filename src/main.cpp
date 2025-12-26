// Lightwatch CLI - Phase 38
// Main entry point for command-line interface

#include <lightwatch/cli.hpp>

int main(int argc, char** argv) {
    return lightwatch::cli::parse_and_run(argc, argv);
}
