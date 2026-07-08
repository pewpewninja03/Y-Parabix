#include <image/bmp_loader.h>

#include <fcntl.h>
#include <unistd.h>
#include <iostream>
#include <stdexcept>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: bmp_header_test <file.bmp>\n";
        return 1;
    }

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) {
        std::perror("open");
        return 1;
    }

    try {
        image::BMPInfo info;
        image::readBMPHeader(fd, info);

        std::cout << "width=" << info.width << "\n";
        std::cout << "height=" << info.height << "\n";
        std::cout << "rowStride=" << info.rowStride << "\n";
        std::cout << "pixelOffset=" << info.pixelOffset << "\n";
        std::cout << "rowsBottomUp=" << info.rowsBottomUp << "\n";

        off_t pos = lseek(fd, 0, SEEK_CUR);
        std::cout << "fd position=" << pos << "\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        close(fd);
        return 2;
    }

    close(fd);
    return 0;
}
