// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_writer.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "pc_sampling_feature.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

using namespace rocprofiler_compute_tool;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// parse_pc_sampling_mode
// ---------------------------------------------------------------------------

TEST(parse_pc_sampling_mode_t, StochasticString)
{
    EXPECT_EQ(parse_pc_sampling_mode("stochastic"), PcSamplingMode::Stochastic);
}

TEST(parse_pc_sampling_mode_t, HostTrapString)
{
    EXPECT_EQ(parse_pc_sampling_mode("host_trap"), PcSamplingMode::HostTrap);
}

TEST(parse_pc_sampling_mode_t, EmptyStringIsDisabled)
{
    EXPECT_EQ(parse_pc_sampling_mode(""), PcSamplingMode::Disabled);
}

TEST(parse_pc_sampling_mode_t, UnknownStringIsDisabled)
{
    EXPECT_EQ(parse_pc_sampling_mode("garbage"), PcSamplingMode::Disabled);
}

// ---------------------------------------------------------------------------
// finalize() — JSON emission + source snapshot
// ---------------------------------------------------------------------------

namespace
{
// A self-contained collector that emits one code object with one instruction
// and reports one source path, so finalize() can be exercised without the real
// disassembler or the collector's mock translator.
class fake_collector_t : public pc_sampling_collector_t
{
public:
    explicit fake_collector_t(std::vector<std::string> source_paths)
        : m_source_paths(std::move(source_paths))
    {
    }

    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t&) override
    {
    }

    void write(code_object_writer_t& writer) override
    {
        writer.start_code_obj(7);
        writer.start_symbol(symbol_t{"vecCopy", 0x10, 0x1000, 1});
        writer.write_instruction(instruction_t{"s_load_b64", "/src/k.hip:42", 0x1000, 0x10, 1});
        writer.end_symbol();
        writer.end_code_obj();
    }

    std::vector<std::string> collect_source_paths() override { return m_source_paths; }

private:
    std::vector<std::string> m_source_paths;
};

class test_pc_sampling_feature_t : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::error_code ec;
        const auto      base = fs::temp_directory_path(ec);
        ASSERT_FALSE(ec);
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        m_tmp = base / (std::string("rpc_pcs_feature_") + info->test_case_name() + "_" + info->name());
        fs::remove_all(m_tmp, ec);
        ASSERT_TRUE(fs::create_directories(m_tmp, ec));
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(m_tmp, ec);
    }

    static void write_file(const fs::path& path, const std::string& content)
    {
        fs::create_directories(path.parent_path());
        std::ofstream out(path);
        out << content;
    }

    fs::path m_tmp;
};
}  // namespace

TEST_F(test_pc_sampling_feature_t, FinalizeWritesJsonAndSnapshotsSource)
{
    // A real source file the collector will report as referenced.
    const fs::path src = m_tmp / "src" / "k.hip";
    write_file(src, "int main() { return 0; }\n");

    const fs::path output = m_tmp / "out" / "ps_file_code_obj_info.json";
    auto collector = std::make_shared<fake_collector_t>(std::vector<std::string>{src.string()});
    pc_sampling_feature_t feature(PcSamplingMode::Stochastic, output, collector);

    feature.finalize();

    // (a) The code-object JSON is written and parseable.
    ASSERT_TRUE(fs::exists(output)) << "expected code-object JSON at " << output;
    std::ifstream  in(output);
    nlohmann::json json;
    ASSERT_NO_THROW(in >> json);
    ASSERT_EQ(json["code_objects"].size(), 1u);
    EXPECT_EQ(json["code_objects"][0]["id"], 7);

    // (b) The source file is snapshotted under <output parent>/code_obj_sources/.
    const fs::path snapshot = output.parent_path() / "code_obj_sources" / src.relative_path();
    EXPECT_TRUE(fs::exists(snapshot)) << "expected source snapshot at " << snapshot;
}

TEST_F(test_pc_sampling_feature_t, FinalizeResolvesRelativeOutputUnderCwd)
{
    // A relative output path resolves against the current directory; the
    // snapshot lands beside it under that same parent.
    const fs::path cwd_before = fs::current_path();
    fs::current_path(m_tmp);

    const fs::path src = m_tmp / "src" / "k.hip";
    write_file(src, "// src\n");

    auto collector = std::make_shared<fake_collector_t>(std::vector<std::string>{src.string()});
    pc_sampling_feature_t feature(PcSamplingMode::Stochastic, "out/ps_file_code_obj_info.json", collector);

    feature.finalize();

    EXPECT_TRUE(fs::exists(m_tmp / "out" / "ps_file_code_obj_info.json"));
    const fs::path snapshot = m_tmp / "out" / "code_obj_sources" / src.relative_path();
    EXPECT_TRUE(fs::exists(snapshot)) << "expected snapshot at " << snapshot;

    fs::current_path(cwd_before);
}
