package com.icaroerasmo.model;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.time.Instant;

@Data
@AllArgsConstructor
public class DetectionRecord {
    private String personName;
    private Double confidence;
    private Instant timestamp;
}
