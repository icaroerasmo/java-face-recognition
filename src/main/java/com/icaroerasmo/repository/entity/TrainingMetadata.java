package com.icaroerasmo.repository.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.Instant;

@Entity
@Table(name = "training_metadata")
@Getter
@Setter
public class TrainingMetadata {

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    private Long id;

    @Column(name = "person_name", nullable = false)
    private String personName;

    @Column(name = "folder_hash", nullable = false)
    private String folderHash;

    @CreationTimestamp
    @Column(name = "first_trained_at", updatable = false)
    private Instant firstTrainedAt;

    @UpdateTimestamp
    @Column(name = "last_trained_at")
    private Instant lastTrainedAt;
}
