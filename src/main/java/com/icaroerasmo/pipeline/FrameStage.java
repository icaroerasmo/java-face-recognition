package com.icaroerasmo.pipeline;

/**
 * A single step of the per-frame pipeline.
 * Stages transfer owned Rects into the {@link FrameContext} and must not
 * deallocate them - the context owns and releases them on close().
 */
public interface FrameStage {

    void process(FrameContext ctx);
}
