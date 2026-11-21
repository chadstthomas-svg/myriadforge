#if defined(_WIN32)
#define MAGIC8BALL_EXPORT __declspec(dllexport)
#else
#define MAGIC8BALL_EXPORT
#endif

#if defined(_MSC_VER)
/* MSVC references this marker when compiling floating-point code without a CRT. */
const int _fltused = 0;
#endif

enum { WIDTH = 7, TABLE_SIZE = 1 << WIDTH };

MAGIC8BALL_EXPORT void magic8ball_table_batch(
    const unsigned char table[TABLE_SIZE], const float facts[][WIDTH],
    unsigned int count, float outputs[][WIDTH])
{
    for (unsigned int row = 0; row < count; ++row) {
        unsigned int mask = 0;
        for (unsigned int column = 0; column < WIDTH; ++column) {
            mask |= (unsigned int)(facts[row][column] != 0.0f) << column;
        }

        unsigned int word = table[mask];
        for (unsigned int column = 0; column < WIDTH; ++column) {
            outputs[row][column] = (float)((word >> column) & 1u);
        }
    }
}
