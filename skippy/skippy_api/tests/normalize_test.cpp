#include "normalize.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>

TEST(EmbeddingNormalization, NormalizesFourElements) {
    const float source[] = {1.0F, -2.0F, 2.0F, -4.0F};
    float output[] = {99.0F, 99.0F, 99.0F, 99.0F};

    skippy::detail::normalize_embedding(source, output, 4);

    EXPECT_FLOAT_EQ(output[0], 0.2F);
    EXPECT_FLOAT_EQ(output[1], -0.4F);
    EXPECT_FLOAT_EQ(output[2], 0.4F);
    EXPECT_FLOAT_EQ(output[3], -0.8F);
}

TEST(EmbeddingNormalization, NormalizesSevenElementsIncludingRemainder) {
    const float source[] = {1.0F, -2.0F, 3.0F, -4.0F, 5.0F, -6.0F, 7.0F};
    float output[] = {99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F, 99.0F};

    skippy::detail::normalize_embedding(source, output, 7);

    const double length = std::sqrt(140.0);
    double squared_norm = 0.0;

    for (std::size_t i = 0; i < 7; ++i) {
        EXPECT_FLOAT_EQ(output[i], static_cast<float>(source[i] / length)) << "at index " << i;

        squared_norm += static_cast<double>(output[i]) * output[i];
    }

    EXPECT_NEAR(squared_norm, 1.0, 1e-6);
}

TEST(EmbeddingNormalization, SupportsInPlaceNormalization) {
    float values[] = {1.0F, -2.0F, 2.0F, -4.0F};

    skippy::detail::normalize_embedding(values, values, 4);

    EXPECT_FLOAT_EQ(values[0], 0.2F);
    EXPECT_FLOAT_EQ(values[1], -0.4F);
    EXPECT_FLOAT_EQ(values[2], 0.4F);
    EXPECT_FLOAT_EQ(values[3], -0.8F);
}

TEST(EmbeddingNormalization, ZeroDimensionsLeavesOutputUntouched) {
    const float source[] = {3.0F};
    float output[] = {99.0F};

    skippy::detail::normalize_embedding(source, output, 0);

    EXPECT_FLOAT_EQ(output[0], 99.0F);
}

TEST(EmbeddingNormalization, ZeroVectorIsCopiedUnchanged) {
    const float source[] = {0.0F, 0.0F, 0.0F};
    float output[] = {99.0F, 99.0F, 99.0F};

    skippy::detail::normalize_embedding(source, output, 3);

    EXPECT_FLOAT_EQ(output[0], 0.0F);
    EXPECT_FLOAT_EQ(output[1], 0.0F);
    EXPECT_FLOAT_EQ(output[2], 0.0F);
}
