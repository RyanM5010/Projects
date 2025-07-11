package com.ecommerce.project.model;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToMany;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.Table;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import java.util.ArrayList;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.ToString;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
@Table(name = "addresses")
public class Address {

  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Long addressId;

  @NotBlank
  @Size(min= 3, message = "Street Name must be at least 3 characters")
  private String street;

  @NotBlank
  @Size(min= 3, message = "Building Name must be at least 3 characters")
  private String buildingName;

  @NotBlank
  @Size(min= 3, message = "City Name must be at least 3 characters")
  private String city;

  @NotBlank
  @Size(min= 2, message = "State Name must be at least 2 characters")
  private String state;

  @NotBlank
  @Size(min= 2, message = "Country Name must be at least 2 characters")
  private String country;

  @NotBlank
  @Size(min= 5, message = "ZipCode must be at least 5 characters")
  private String zipCode;

  @ManyToOne
  @JoinColumn(name = "user_id")
  private User user;

  public Address(String street, String buildingName, String city, String state, String country,
      String zipCode, User user) {
    this.street = street;
    this.buildingName = buildingName;
    this.city = city;
    this.state = state;
    this.country = country;
    this.zipCode = zipCode;
    this.user = user;
  }
}
