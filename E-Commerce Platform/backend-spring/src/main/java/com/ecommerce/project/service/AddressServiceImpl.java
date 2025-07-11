package com.ecommerce.project.service;


import com.ecommerce.project.exceptions.ResourceNotFoundException;
import com.ecommerce.project.model.Address;
import com.ecommerce.project.model.User;
import com.ecommerce.project.payload.AddressDTO;
import com.ecommerce.project.reprositories.AddressRepository;
import com.ecommerce.project.reprositories.UserRepository;
import java.util.List;
import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class AddressServiceImpl implements AddressService {

  @Autowired
  ModelMapper modelMapper;

  @Autowired
  AddressRepository addressRepository;

  @Autowired
  UserRepository userRepository;

  @Override
  public AddressDTO createAddress(AddressDTO addressDTO, User user) {
    Address address = modelMapper.map(addressDTO, Address.class);
    List<Address> addressList = user.getAddresses();
    addressList.add(address);
    user.setAddresses(addressList);
    address.setUser(user);
    Address savedAddress = addressRepository.save(address);

    return modelMapper.map(savedAddress, AddressDTO.class);
  }

  @Override
  public List<AddressDTO> getAddresses() {
    List<Address> addressList = addressRepository.findAll();
    return addressList.stream()
        .map(address -> modelMapper.map(address, AddressDTO.class))
        .toList();
  }

  @Override
  public AddressDTO getAddressesById(Long addressId) {
    Address address = addressRepository.findById(addressId)
        .orElseThrow( () -> new ResourceNotFoundException("Address", "addressId", addressId));
    return modelMapper.map(address, AddressDTO.class);
  }

  @Override
  public List<AddressDTO> getUserAddresses(User user) {
    List<Address> addressList = user.getAddresses();
    return addressList.stream()
        .map(address -> modelMapper.map(address, AddressDTO.class))
        .toList();
  }

  @Override
  public AddressDTO updateAddressesById(Long addressId, AddressDTO addressDTO) {
    Address addressFromDB = addressRepository.findById(addressId)
        .orElseThrow( () -> new ResourceNotFoundException("Address", "addressId", addressId));

    addressFromDB.setCity(addressDTO.getCity());
    addressFromDB.setState(addressDTO.getState());
    addressFromDB.setZipCode(addressDTO.getZipCode());
    addressFromDB.setCountry(addressDTO.getCountry());
    addressFromDB.setStreet(addressDTO.getStreet());
    addressFromDB.setBuildingName(addressDTO.getBuildingName());

    Address updatedAddress = addressRepository.save(addressFromDB);

    User user = addressFromDB.getUser();
    user.getAddresses().removeIf(address -> address.getAddressId().equals(addressId));
    user.getAddresses().add(updatedAddress);
    userRepository.save(user);
    return modelMapper.map(updatedAddress, AddressDTO.class);
  }

  @Override
  public String deleteAddressesById(Long addressId) {
    Address addressfromDB = addressRepository.findById(addressId)
        .orElseThrow( () -> new ResourceNotFoundException("Address", "addressId", addressId));
    User user = addressfromDB.getUser();
    user.getAddresses().removeIf(address -> address.getAddressId().equals(addressId));
    userRepository.save(user);
    addressRepository.delete(addressfromDB);
    return "Address successfully deleted with id: " + addressId;
  }
}
