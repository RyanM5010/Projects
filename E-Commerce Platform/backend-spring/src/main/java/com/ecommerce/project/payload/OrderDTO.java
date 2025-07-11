package com.ecommerce.project.payload;

import java.time.LocalDateTime;
import java.util.List;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class OrderDTO {
  private Long orderId;
  private String email;
  private List<OrderItemDTO> orderItems;
  private LocalDateTime orderDate;
  private PaymentDTO payment;
  private Double totalAmount;
  private String orderStatus;
  private Long addressId;
}
