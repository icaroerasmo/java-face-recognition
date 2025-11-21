package com.icaroerasmo.repository.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

@Entity
@Table(name = "trained_dataset")
@Getter
@Setter
public class TrainedDataset {

    @Id
    @Column(name = "id")
    private Long id = 1L;

    @Column(name = "model_xml", nullable = false, columnDefinition = "BLOB")
    private byte[] modelXml;
}
