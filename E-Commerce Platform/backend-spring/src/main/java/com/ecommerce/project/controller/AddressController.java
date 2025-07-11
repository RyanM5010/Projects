package com.ecommerce.project.controller;


import com.ecommerce.project.model.User;
import com.ecommerce.project.payload.AddressDTO;
import com.ecommerce.project.service.AddressService;
import com.ecommerce.project.util.AuthUtil;
import jakarta.validation.Valid;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
public class AddressController {

  @Autowired
  private AddressService addressService;
  @Autowired
  private AuthUtil authUtil;


  @PostMapping("/addresses")
  public ResponseEntity<AddressDTO> createAddress(@Valid @RequestBody AddressDTO addressDTO) {
    User user = authUtil.loggedInUser();
    AddressDTO savedAddressDTO = addressService.createAddress(addressDTO, user);
    return new ResponseEntity<>(savedAddressDTO, HttpStatus.CREATED);
  }

  @GetMapping("/addresses")
  public ResponseEntity<List<AddressDTO>> getAllAddress() {
    List<AddressDTO> addressDTOList = addressService.getAddresses();
    return new ResponseEntity<>(addressDTOList, HttpStatus.OK);
  }

  @GetMapping("/addresses/{addressId}")
  public ResponseEntity<AddressDTO> getAddressById(@PathVariable Long addressId) {
    AddressDTO addressDTO = addressService.getAddressesById(addressId);
    return new ResponseEntity<>(addressDTO, HttpStatus.OK);
  }

  @GetMapping("/users/addresses")
  public ResponseEntity<List<AddressDTO>> getUserAddress() {
    User user = authUtil.loggedInUser();
    List<AddressDTO> addressDTOList = addressService.getUserAddresses(user);
    return new ResponseEntity<>(addressDTOList, HttpStatus.OK);
  }


  @PutMapping("/addresses/{addressId}")
  public ResponseEntity<AddressDTO> updateAddressById(@PathVariable Long addressId, @RequestBody AddressDTO addressDTO) {
    AddressDTO updatedAddressDTO = addressService.updateAddressesById(addressId, addressDTO);
    return new ResponseEntity<>(updatedAddressDTO, HttpStatus.OK);
  }


  @DeleteMapping("/addresses/{addressId}")
  public ResponseEntity<String> deleteAddressById(@PathVariable Long addressId) {
    String status = addressService.deleteAddressesById(addressId);
    return new ResponseEntity<>(status, HttpStatus.OK);
  }











}
