#include "runtime.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <expected>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using skippy::Model;
using skippy::ModelConfig;
using skippy::NativeRuntime;
using skippy::RuntimeError;
using skippy::Session;
using skippy::SessionConfig;

using ::testing::_;
using ::testing::ByMove;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Ref;
using ::testing::Return;
using ::testing::StrictMock;

using DestructionLog = std::vector<std::string>;

class LoggingModelResource {
public:
    LoggingModelResource(DestructionLog& log, std::string name)
        : log_(log), name_(std::move(name)) {}
    ~LoggingModelResource() { log_.push_back("model:" + name_); }
    LoggingModelResource(const LoggingModelResource&) = delete;
    LoggingModelResource& operator=(const LoggingModelResource&) = delete;

private:
    DestructionLog& log_;
    std::string name_;
};

class LoggingSessionResource {
public:
    LoggingSessionResource(DestructionLog& log, std::string name)
        : log_(log), name_(std::move(name)) {}
    ~LoggingSessionResource() { log_.push_back("session:" + name_); }
    LoggingSessionResource(const LoggingSessionResource&) = delete;
    LoggingSessionResource& operator=(const LoggingSessionResource&) = delete;

private:
    DestructionLog& log_;
    std::string name_;
};

using ModelResult = std::expected<std::unique_ptr<LoggingModelResource>, RuntimeError>;
using SessionResult = std::expected<std::unique_ptr<LoggingSessionResource>, RuntimeError>;

class MockBackend {
public:
    using ModelResource = LoggingModelResource;
    using SessionResource = LoggingSessionResource;

    MOCK_METHOD(ModelResult, load_model, (const ModelConfig&));
    MOCK_METHOD(SessionResult, create_session, (LoggingModelResource&, const SessionConfig&));
};
static_assert(skippy::RuntimeBackend<StrictMock<MockBackend>>);

class NativeRuntimeTest : public ::testing::Test {
protected:
    DestructionLog log_;
    NativeRuntime<StrictMock<MockBackend>> runtime_{std::in_place};
    StrictMock<MockBackend>& backend_ = runtime_.backend();

    ModelConfig model_config_{.path = "model.gguf"};
    SessionConfig session_config_{};

    LoggingModelResource& arm_load_model(std::string name) {
        auto resource = std::make_unique<LoggingModelResource>(log_, std::move(name));
        auto& view = *resource;
        EXPECT_CALL(backend_, load_model(_))
            .WillOnce(Return(ByMove(ModelResult{std::move(resource)})))
            .RetiresOnSaturation();
        return view;
    }

    LoggingSessionResource& arm_create_session(LoggingModelResource& model, std::string name) {
        auto resource = std::make_unique<LoggingSessionResource>(log_, std::move(name));
        auto& view = *resource;
        EXPECT_CALL(backend_, create_session(Ref(model), _))
            .WillOnce(Return(ByMove(SessionResult{std::move(resource)})))
            .RetiresOnSaturation();
        return view;
    }

    std::expected<Model, RuntimeError> load_model() { return runtime_.load_model(model_config_); }

    std::expected<Session, RuntimeError> create_session(const Model& model) {
        return runtime_.create_session(model, session_config_);
    }
};

TEST_F(NativeRuntimeTest, LoadModelReturnsHandleForBackendResource) {
    arm_load_model("a");

    const auto model = load_model();

    ASSERT_TRUE(model.has_value());
    EXPECT_THAT(log_, IsEmpty());
}

TEST_F(NativeRuntimeTest, LoadModelPropagatesBackendError) {
    EXPECT_CALL(backend_, load_model(_))
        .WillOnce(Return(ByMove(ModelResult{std::unexpected(RuntimeError::LoadFailed)})));

    const auto model = load_model();

    ASSERT_FALSE(model.has_value());
    EXPECT_EQ(model.error(), RuntimeError::LoadFailed);
}

TEST_F(NativeRuntimeTest, LoadModelRejectsNullResourceFromBackend) {
    EXPECT_CALL(backend_, load_model(_))
        .WillOnce(Return(ByMove(ModelResult{std::unique_ptr<LoggingModelResource>{}})));

    const auto model = load_model();

    ASSERT_FALSE(model.has_value());
    EXPECT_EQ(model.error(), RuntimeError::InvalidBackendResult);
}

TEST_F(NativeRuntimeTest, CreateSessionHandsBackendTheModelResource) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());

    // Ref(model_resource) inside arm_create_session() is the assertion.
    arm_create_session(model_resource, "s");

    const auto session = create_session(*model);

    ASSERT_TRUE(session.has_value());
    EXPECT_THAT(log_, IsEmpty());
}

TEST_F(NativeRuntimeTest, CreateSessionPropagatesBackendError) {
    arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    EXPECT_CALL(backend_, create_session(_, _))
        .WillOnce(
            Return(ByMove(SessionResult{std::unexpected(RuntimeError::SessionCreationFailed)})));

    const auto session = create_session(*model);

    ASSERT_FALSE(session.has_value());
    EXPECT_EQ(session.error(), RuntimeError::SessionCreationFailed);
}

TEST_F(NativeRuntimeTest, CreateSessionRejectsNullResourceFromBackend) {
    arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    EXPECT_CALL(backend_, create_session(_, _))
        .WillOnce(Return(ByMove(SessionResult{std::unique_ptr<LoggingSessionResource>{}})));

    const auto session = create_session(*model);

    ASSERT_FALSE(session.has_value());
    EXPECT_EQ(session.error(), RuntimeError::InvalidBackendResult);
}

TEST_F(NativeRuntimeTest, CreateSessionRejectsUnloadedModel) {
    arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    ASSERT_FALSE(runtime_.unload_model(*model).has_value());

    // No create_session expectation: the backend must not be consulted.
    const auto session = create_session(*model);

    ASSERT_FALSE(session.has_value());
    EXPECT_EQ(session.error(), RuntimeError::InvalidModel);
}

TEST_F(NativeRuntimeTest, CreateSessionRejectsHandleFromAnotherRuntime) {
    // Slot 0 is live in both runtimes; only the generation tells them apart.
    arm_load_model("local");
    ASSERT_TRUE(load_model().has_value());

    DestructionLog other_log;
    NativeRuntime<StrictMock<MockBackend>> other{std::in_place};
    EXPECT_CALL(other.backend(), load_model(_))
        .WillOnce(Return(
            ByMove(ModelResult{std::make_unique<LoggingModelResource>(other_log, "foreign")})));
    auto foreign = other.load_model(model_config_);
    ASSERT_TRUE(foreign.has_value());

    const auto session = create_session(*foreign);

    ASSERT_FALSE(session.has_value());
    EXPECT_EQ(session.error(), RuntimeError::InvalidModel);
}

TEST_F(NativeRuntimeTest, MovedFromModelHandleIsRejectedAndMoveTargetWorks) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());

    Model moved = std::move(*model);

    const auto via_source = create_session(*model);
    ASSERT_FALSE(via_source.has_value());
    EXPECT_EQ(via_source.error(), RuntimeError::InvalidModel);
    EXPECT_EQ(*runtime_.unload_model(*model), RuntimeError::InvalidModel);
    EXPECT_THAT(log_, IsEmpty());

    arm_create_session(model_resource, "s");
    EXPECT_TRUE(create_session(moved).has_value());
}

TEST_F(NativeRuntimeTest, MovedFromSessionHandleIsRejected) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    arm_create_session(model_resource, "s");
    auto session = create_session(*model);
    ASSERT_TRUE(session.has_value());

    Session moved = std::move(*session);

    EXPECT_EQ(*runtime_.release_session(*session), RuntimeError::InvalidSession);
    EXPECT_THAT(log_, IsEmpty());

    EXPECT_FALSE(runtime_.release_session(moved).has_value());
    EXPECT_THAT(log_, ElementsAre("session:s"));
}

TEST_F(NativeRuntimeTest, WithModelBorrowsTheExactResource) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());

    const auto seen = runtime_.with_model(*model, [](LoggingModelResource& m) { return &m; });

    ASSERT_TRUE(seen.has_value());
    EXPECT_EQ(*seen, &model_resource);
}

TEST_F(NativeRuntimeTest, WithSessionBorrowsTheExactResource) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    auto& session_resource = arm_create_session(model_resource, "s");
    auto session = create_session(*model);
    ASSERT_TRUE(session.has_value());

    const auto seen = runtime_.with_session(*session, [](LoggingSessionResource& s) { return &s; });

    ASSERT_TRUE(seen.has_value());
    EXPECT_EQ(*seen, &session_resource);
}

TEST_F(NativeRuntimeTest, WithSessionSupportsVoidVisitors) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    arm_create_session(model_resource, "s");
    auto session = create_session(*model);
    ASSERT_TRUE(session.has_value());
    int calls = 0;

    const std::expected<void, RuntimeError> result =
        runtime_.with_session(*session, [&](LoggingSessionResource&) { ++calls; });

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(calls, 1);
}

TEST_F(NativeRuntimeTest, WithSessionDoesNotInvokeVisitorForInvalidHandle) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    arm_create_session(model_resource, "s");
    auto session = create_session(*model);
    ASSERT_TRUE(session.has_value());
    ASSERT_FALSE(runtime_.release_session(*session).has_value());
    int calls = 0;

    const auto result =
        runtime_.with_session(*session, [&](LoggingSessionResource&) { return ++calls; });

    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), RuntimeError::InvalidSession);
    EXPECT_EQ(calls, 0);
}

TEST_F(NativeRuntimeTest, ConstWithModelSeesConstResource) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    const auto& const_runtime = runtime_;

    const auto seen =
        const_runtime.with_model(*model, [](const LoggingModelResource& m) { return &m; });

    ASSERT_TRUE(seen.has_value());
    EXPECT_EQ(*seen, &model_resource);
}

TEST_F(NativeRuntimeTest, ReleaseSessionDestroysResourceAndClearsHandle) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    arm_create_session(model_resource, "s");
    auto session = create_session(*model);
    ASSERT_TRUE(session.has_value());

    ASSERT_FALSE(runtime_.release_session(*session).has_value());
    EXPECT_THAT(log_, ElementsAre("session:s"));

    const auto again = runtime_.release_session(*session);
    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(*again, RuntimeError::InvalidSession);
    EXPECT_THAT(log_, ElementsAre("session:s"));
}

TEST_F(NativeRuntimeTest, ReleaseSessionLeavesOtherSessionsAlone) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    arm_create_session(model_resource, "s1");
    auto s1 = create_session(*model);
    ASSERT_TRUE(s1.has_value());
    arm_create_session(model_resource, "s2");
    auto s2 = create_session(*model);
    ASSERT_TRUE(s2.has_value());

    ASSERT_FALSE(runtime_.release_session(*s1).has_value());

    EXPECT_THAT(log_, ElementsAre("session:s1"));
    EXPECT_FALSE(runtime_.release_session(*s2).has_value());
    EXPECT_THAT(log_, ElementsAre("session:s1", "session:s2"));
}

TEST_F(NativeRuntimeTest, UnloadModelRefusesWhileSessionIsLive) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    arm_create_session(model_resource, "s");
    auto session = create_session(*model);
    ASSERT_TRUE(session.has_value());

    const auto refused = runtime_.unload_model(*model);
    ASSERT_TRUE(refused.has_value());
    EXPECT_EQ(*refused, RuntimeError::ModelInUse);
    EXPECT_THAT(log_, IsEmpty()); // neither resource was touched

    ASSERT_FALSE(runtime_.release_session(*session).has_value());
    ASSERT_FALSE(runtime_.unload_model(*model).has_value());
    EXPECT_THAT(log_, ElementsAre("session:s", "model:a"));
}

TEST_F(NativeRuntimeTest, UnloadModelIgnoresSessionsOfOtherModels) {
    arm_load_model("a");
    auto a = load_model();
    ASSERT_TRUE(a.has_value());
    auto& b_resource = arm_load_model("b");
    auto b = load_model();
    ASSERT_TRUE(b.has_value());
    arm_create_session(b_resource, "sb");
    ASSERT_TRUE(create_session(*b).has_value());

    EXPECT_FALSE(runtime_.unload_model(*a).has_value());

    EXPECT_THAT(log_, ElementsAre("model:a"));
}

TEST_F(NativeRuntimeTest, UnloadModelInvalidatesHandle) {
    arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    ASSERT_FALSE(runtime_.unload_model(*model).has_value());

    const auto again = runtime_.unload_model(*model);

    ASSERT_TRUE(again.has_value());
    EXPECT_EQ(*again, RuntimeError::InvalidModel);
    EXPECT_THAT(log_, ElementsAre("model:a")); // destroyed exactly once
}

TEST_F(NativeRuntimeTest, UnloadedSlotIsReusedUnderNewGeneration) {
    arm_load_model("first");
    auto first = load_model();
    ASSERT_TRUE(first.has_value());
    ASSERT_FALSE(runtime_.unload_model(*first).has_value());

    auto& second_resource = arm_load_model("second");
    auto second = load_model();
    ASSERT_TRUE(second.has_value());

    EXPECT_EQ(create_session(*first).error(), RuntimeError::InvalidModel);
    arm_create_session(second_resource, "s");
    EXPECT_TRUE(create_session(*second).has_value());
}

TEST_F(NativeRuntimeTest, ShutdownOnEmptyRuntimeReleasesNothing) {
    const auto summary = runtime_.shutdown();

    EXPECT_EQ(summary.sessions_released, 0U);
    EXPECT_EQ(summary.models_released, 0U);
    EXPECT_THAT(log_, IsEmpty());
}

TEST_F(NativeRuntimeTest, ShutdownDestroysEverySessionBeforeAnyModel) {
    auto& a_resource = arm_load_model("a");
    auto a = load_model();
    ASSERT_TRUE(a.has_value());
    auto& b_resource = arm_load_model("b");
    auto b = load_model();
    ASSERT_TRUE(b.has_value());
    arm_create_session(a_resource, "sa");
    ASSERT_TRUE(create_session(*a).has_value());
    arm_create_session(b_resource, "sb");
    ASSERT_TRUE(create_session(*b).has_value());

    const auto summary = runtime_.shutdown();

    EXPECT_EQ(summary.sessions_released, 2U);
    EXPECT_EQ(summary.models_released, 2U);
    EXPECT_THAT(log_, ElementsAre("session:sa", "session:sb", "model:a", "model:b"));
}

TEST_F(NativeRuntimeTest, ShutdownInvalidatesOutstandingHandles) {
    auto& model_resource = arm_load_model("a");
    auto model = load_model();
    ASSERT_TRUE(model.has_value());
    arm_create_session(model_resource, "s");
    auto session = create_session(*model);
    ASSERT_TRUE(session.has_value());

    runtime_.shutdown();

    EXPECT_EQ(*runtime_.release_session(*session), RuntimeError::InvalidSession);
    EXPECT_EQ(*runtime_.unload_model(*model), RuntimeError::InvalidModel);
    EXPECT_THAT(log_, ElementsAre("session:s", "model:a")); // nothing destroyed twice
}

TEST_F(NativeRuntimeTest, ShutdownIsIdempotentAndLeavesRuntimeUsable) {
    arm_load_model("a");
    ASSERT_TRUE(load_model().has_value());
    runtime_.shutdown();

    const auto second = runtime_.shutdown();
    EXPECT_EQ(second.sessions_released, 0U);
    EXPECT_EQ(second.models_released, 0U);

    arm_load_model("b");
    EXPECT_TRUE(load_model().has_value());
    EXPECT_THAT(log_, ElementsAre("model:a"));
}

TEST(NativeRuntimeDestruction, DestructorDestroysSessionsThenModels) {
    DestructionLog log;
    {
        NativeRuntime<StrictMock<MockBackend>> runtime{std::in_place};
        EXPECT_CALL(runtime.backend(), load_model(_))
            .WillOnce(
                Return(ByMove(ModelResult{std::make_unique<LoggingModelResource>(log, "a")})));
        EXPECT_CALL(runtime.backend(), create_session(_, _))
            .WillOnce(
                Return(ByMove(SessionResult{std::make_unique<LoggingSessionResource>(log, "s")})));
        auto model = runtime.load_model(ModelConfig{});
        ASSERT_TRUE(model.has_value());
        ASSERT_TRUE(runtime.create_session(*model).has_value());
        EXPECT_THAT(log, IsEmpty());
    }
    EXPECT_THAT(log, ElementsAre("session:s", "model:a"));
}

TEST(NativeRuntimeDestruction, ReleasedResourcesAreNotDestroyedAgain) {
    DestructionLog log;
    {
        NativeRuntime<StrictMock<MockBackend>> runtime{std::in_place};
        EXPECT_CALL(runtime.backend(), load_model(_))
            .WillOnce(
                Return(ByMove(ModelResult{std::make_unique<LoggingModelResource>(log, "a")})));
        auto model = runtime.load_model(ModelConfig{});
        ASSERT_TRUE(model.has_value());
        ASSERT_FALSE(runtime.unload_model(*model).has_value());
        EXPECT_THAT(log, ElementsAre("model:a"));
    }
    EXPECT_THAT(log, ElementsAre("model:a"));
}

struct CountingBackend {
    struct ModelResource {};
    struct SessionResource {};
    int seed = 0;

    static std::expected<std::unique_ptr<ModelResource>, RuntimeError>
    load_model(const ModelConfig& /*unused*/) {
        return std::make_unique<ModelResource>();
    }
    static std::expected<std::unique_ptr<SessionResource>, RuntimeError>
    create_session(ModelResource& /*unused*/, const SessionConfig& /*unused*/) {
        return std::make_unique<SessionResource>();
    }
};
static_assert(skippy::RuntimeBackend<CountingBackend>);

TEST(NativeRuntimeConstruction, InPlaceForwardsArgumentsToBackend) {
    NativeRuntime<CountingBackend> runtime{std::in_place, CountingBackend{.seed = 7}};
    EXPECT_EQ(runtime.backend().seed, 7);
}

TEST(NativeRuntimeConstruction, MovesAReadyBackendIn) {
    CountingBackend backend{.seed = 11};
    NativeRuntime<CountingBackend> runtime{backend};
    EXPECT_EQ(runtime.backend().seed, 11);
}

} // namespace
