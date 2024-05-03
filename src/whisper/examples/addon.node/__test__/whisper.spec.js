const path = require("path");
const { whisper } = require(path.join(
  __dirname,
  "../../../build/Release/addon.node"
));
const { promisify } = require("util");

const whisperAsync = promisify(whisper);

const whisperParamsMock = {
  language: "en",
  model: path.join(__dirname, "../../../models/ggml-base.en.bin"),
  fname_inp: path.join(__dirname, "../../../samples/jfk.wav"),
  use_gpu: true,
  no_timestamps: false,
};

describe("Run whisper.node", () => {
    test("it should receive a non-empty value", async () => {
        let result = await whisperAsync(whisperParamsMock);

        expect(result.length).toBeGreaterThan(0);
    }, 10000);
});

