test(
  'ut_network',
  executable(
    'ut_network',
    'network.cc',
    include_directories: inc,
    dependencies: test_dep
  )
)

# layer
test(
  'ut_input_layer',
  executable(
    'ut_input_layer',
    'layer/input_layer.cc',
    include_directories: inc,
    dependencies: test_dep
  )
)

# type_traits
test(
  'ut_connect_layers',
  executable(
    'ut_connect_layers',
    'type_traits/connect_layers.cc',
    include_directories: inc,
    dependencies: test_dep
  )
)

test(
  'ut_remove_input_layer',
  executable(
    'ut_remove_input_layer',
    'type_traits/remove_input_layer.cc',
    include_directories: inc,
    dependencies: test_dep
  )
)

# concepts
test(
  'ut_connectable_layers',
  executable(
    'ut_connectable_layers',
    'concepts/connectable_layers.cc',
    include_directories: inc,
    dependencies: test_dep
  )
)

test(
  'ut_layer',
  executable(
    'ut_layer',
    'concepts/layer.cc',
    include_directories: inc,
    dependencies: test_dep
  )
)

