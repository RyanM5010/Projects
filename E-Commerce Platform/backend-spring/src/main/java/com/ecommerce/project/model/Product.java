package com.ecommerce.project.model;

import com.fasterxml.jackson.annotation.JsonEnumDefaultValue;
import jakarta.persistence.CascadeType;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
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
@Table(name = "products")
@ToString
public class Product {

  @Id
  @GeneratedValue(strategy = GenerationType.AUTO)
  private Long productId;
  @NotBlank
  @Size(min = 3, message = "Product name must at least contains 3 characters")
  private String productName;
  @NotBlank
  @Size(min = 6, message = "Product name must at least contains 6 characters")
  private String description;
  private String image;
  private double price;
  private double specialPrice;
  private double discount;
  private Integer quantity;

  @ManyToOne
  @JoinColumn(name = "category_id")
  private Category category;

  @ManyToOne
  @JoinColumn(name = "seller_id")
  private User user;

  @OneToMany(mappedBy = "product", cascade = {CascadeType.PERSIST, CascadeType.MERGE}, fetch = FetchType.EAGER)
  private List<CartItem> products = new ArrayList<>();

}
